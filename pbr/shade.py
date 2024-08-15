import os
from typing import Dict, Optional, Union

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from .light import CubemapLight
from utils.general_utils import safe_normalize, reflect, dot
from . import renderutils as ru


# Lazarov 2013, "Getting More Physical in Call of Duty: Black Ops II"
# https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
def envBRDF_approx(roughness: torch.Tensor, NoV: torch.Tensor) -> torch.Tensor:
    c0 = torch.tensor([-1.0, -0.0275, -0.572, 0.022], device=roughness.device)
    c1 = torch.tensor([1.0, 0.0425, 1.04, -0.04], device=roughness.device)
    c2 = torch.tensor([-1.04, 1.04], device=roughness.device)
    r = roughness * c0 + c1
    a004 = (
        torch.minimum(torch.pow(r[..., (0,)], 2), torch.exp2(-9.28 * NoV)) * r[..., (0,)]
        + r[..., (1,)]
    )
    AB = (a004 * c2 + r[..., 2:]).clamp(min=0.0, max=1.0)
    return AB


def saturate_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True).clamp(min=1e-4, max=1.0)


# Tone Mapping
def aces_film(rgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    EPS = 1e-6
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    rgb = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)
    if isinstance(rgb, np.ndarray):
        return rgb.clip(min=0.0, max=1.0)
    elif isinstance(rgb, torch.Tensor):
        return rgb.clamp(min=0.0, max=1.0)


def linear_to_srgb(linear: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError


def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055
    )


def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = (
        torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1)
        if f.shape[-1] == 4
        else _rgb_to_srgb(f)
    )
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4)
    )


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = (
        torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1)
        if f.shape[-1] == 4
        else _srgb_to_rgb(f)
    )
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


def get_brdf_lut() -> torch.Tensor:
    brdf_lut_path = os.path.join(os.path.dirname(__file__), "brdf_256_256.bin")
    brdf_lut = torch.from_numpy(
        np.fromfile(brdf_lut_path, dtype=np.float32).reshape(1, 256, 256, 2)
    )
    return brdf_lut


def pbr_shading(
    light: CubemapLight,
    normals: torch.Tensor,  # [H, W, 3]
    view_dirs: torch.Tensor,  # [H, W, 3]
    albedo: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
    mask: torch.Tensor,  # [H, W, 1]
    tone: bool = False,
    gamma: bool = False,
    occlusion: Optional[torch.Tensor] = None,  # [H, W, 1]
    irradiance: Optional[torch.Tensor] = None,  # [H, W, 1]
    metallic: Optional[torch.Tensor] = None,
    brdf_lut: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
) -> Dict:
    H, W, _ = normals.shape
    if background is None:
        background = torch.zeros_like(normals)  # [H, W, 3]

    # prepare
    normals = normals.reshape(1, H, W, 3)
    view_dirs = view_dirs.reshape(1, H, W, 3)
    albedo = albedo.reshape(1, H, W, 3)
    roughness = roughness.reshape(1, H, W, 1)

    results = {}
    # prepare
    ref_dirs = (
        2.0 * (normals * view_dirs).sum(-1, keepdims=True).clamp(min=0.0) * normals - view_dirs
    )  # [1, H, W, 3]

    # Diffuse lookup
    diffuse_light = dr.texture(
        light.diffuse[None, ...],  # [1, 6, 16, 16, 3]
        normals.contiguous(),  # [1, H, W, 3]
        filter_mode="linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]

    if occlusion is not None:
        diffuse_light = diffuse_light * occlusion[None] + (1 - occlusion[None]) * irradiance[None]

    results["diffuse_light"] = diffuse_light[0]
    diffuse_rgb = diffuse_light * albedo  # [1, H, W, 3]

    # specular
    NoV = saturate_dot(normals, view_dirs)  # [1, H, W, 1]
    fg_uv = torch.cat((NoV, roughness), dim=-1)  # [1, H, W, 2]
    fg_lookup = dr.texture(
        brdf_lut,  # [1, 256, 256, 2]
        fg_uv.contiguous(),  # [1, H, W, 2]
        filter_mode="linear",
        boundary_mode="clamp",
    )  # [1, H, W, 2]

    # Roughness adjusted specular env lookup
    miplevel = light.get_mip(roughness)  # [1, H, W, 1]
    spec = dr.texture(
        light.specular[0][None, ...],  # [1, 6, env_res, env_res, 3]
        ref_dirs.contiguous(),  # [1, H, W, 3]
        mip=list(m[None, ...] for m in light.specular[1:]),
        mip_level_bias=miplevel[..., 0],  # [1, H, W]
        filter_mode="linear-mipmap-linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]

    # Compute aggregate lighting
    if metallic is None:
        F0 = torch.ones_like(albedo) * 0.04  # [1, H, W, 3]
    else:
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic
    reflectance = F0 * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]  # [1, H, W, 3]
    specular_rgb = spec * reflectance  # [1, H, W, 3]

    render_rgb = diffuse_rgb + specular_rgb  # [1, H, W, 3]

    render_rgb = render_rgb.squeeze()  # [H, W, 3]

    if tone:
        # Tone Mapping
        render_rgb = aces_film(render_rgb)
    else:
        render_rgb = render_rgb.clamp(min=0.0, max=1.0)

    ### NOTE: close `gamma` will cause better resuls in novel view synthesis but wrose relighting results.
    ### NOTE: it is worth to figure out a better way to handle both novel view synthesis and relighting
    if gamma:
        render_rgb = linear_to_srgb(render_rgb.squeeze())

    render_rgb = torch.where(mask, render_rgb, background)

    results.update(
        {
            "render_rgb": render_rgb,
            "diffuse_rgb": diffuse_rgb.squeeze(),
            "specular_rgb": specular_rgb.squeeze(),
        }
    )

    return results


def pbr_shading_2dgs(light, normals, wo, wi, albedo, roughness, metallic, brdf_lut):
    
    # 2DGS-PBR With GSIR shading [minibatch_size=1, height=1, width=num_G, C=3] 
    # a pic of size (1, num_G)

    results = {}
    diffuse_light = dr.texture(
        light.diffuse[None, ...],  # [1, 6, 16, 16, 3]
        normals.contiguous(),
        filter_mode="linear",
        boundary_mode="cube",
    ).squeeze() # [numG, 3]

    #ignore indirect illumination for now
    # if occlusion is not None:
    #   diffuse_light = diffuse_light * occlusion[None] + (1 - occlusion[None]) * irradiance[None]

    diffuse_rgb = albedo * diffuse_light # [numG, 3]
    NoV = saturate_dot(normals.squeeze(), wo)
    fg_uv = torch.cat((NoV, roughness), dim=-1)[None,None,:,:] #[1,1,numG,2]
    fg_lookup = dr.texture(
        brdf_lut,  # [1, 256, 256, 2]
        fg_uv.contiguous(), #[1,1,numG,2]
        filter_mode="linear",
        boundary_mode="clamp",
    ).squeeze() # [numG, 2] #用这种方式查询会不会有超出内存？

    miplevel = light.get_mip(roughness) # [numG, 1]
    spec = dr.texture(
        light.specular[0][None, ...],  # [1, 6, 256, 256, 3]
        wi.contiguous(),
        mip=list(m[None, ...] for m in light.specular[1:]),
        mip_level_bias=miplevel[:,:,None].permute(1,2,0).contiguous(),#[1,1,numG]
        filter_mode="linear-mipmap-linear",
        boundary_mode="cube",
    ).squeeze() # [numG, 3]

    if metallic is None:
        F0 = torch.ones_like(albedo) * 0.04  # [numG, 3]
    else:
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic
    reflectance = F0 * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]  # [numG, 3]
    specular_rgb = spec * reflectance

    rgb = diffuse_rgb + specular_rgb
    results["rgb"] = rgb
    results["diffuse"] = diffuse_rgb
    results["specular"] = specular_rgb    
    return results

def pbr_shade_2dgs_gshader(light, gb_normal, kd, ks, kr, wo, reflvec, brdf_lut, specular=True):
    # (1, 1, num_G, C)
    if specular:
        diffuse_raw = kd
        roughness = kr
        spec_col  = ks
        diff_col  = 1.0 - ks
    else:
        raise NotImplementedError

    # reflvec = safe_normalize(reflect(wo, gb_normal))
    nrmvec = gb_normal
    if light.mtx is not None: # Rotate lookup
        mtx = torch.as_tensor(light.mtx, dtype=torch.float32, device='cuda')
        reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
        nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)

    ambient = dr.texture(light.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
    # specular_linear = ambient * specular_tint
    specular_linear = ambient * diff_col

    if specular:
        # Lookup FG term from lookup texture
        NdotV = torch.clamp(dot(wo, gb_normal.squeeze()), min=1e-4)
        fg_uv = torch.cat((NdotV, roughness), dim=-1)[None,None,:,:]
        # if not hasattr(light, '_FG_LUT'):
        #     light._FG_LUT = torch.as_tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
        fg_lookup = dr.texture(brdf_lut, fg_uv.contiguous(), filter_mode='linear', boundary_mode='clamp')

        # Roughness adjusted specular env lookup
        miplevel = light.get_mip(roughness)
        spec = dr.texture(light.specular[0][None, ...], 
                          reflvec.contiguous(), 
                          mip=list(m[None, ...] for m in light.specular[1:]), 
                          mip_level_bias=miplevel[:,:,None].permute(1,2,0).contiguous(), #[1,1,numG]
                          filter_mode='linear-mipmap-linear', 
                          boundary_mode='cube')

        # Compute aggregate lighting
        # reflectance = specular_tint * fg_lookup[...,0:1] + fg_lookup[...,1:2]
        reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
        specular_linear += spec * reflectance
    extras = {"specular": specular_linear}

    diffuse_linear = torch.sigmoid(diffuse_raw - np.log(3.0))
    extras["diffuse"] = diffuse_linear

    rgb = specular_linear + diffuse_linear

    return rgb, extras

def gshader_deferred_shading(
    light: CubemapLight,
    normals: torch.Tensor,  # [H, W, 3]
    view_dirs: torch.Tensor,  # [H, W, 3]
    kd: torch.Tensor,  # [H, W, 3]
    kr: torch.Tensor,  # [W, 1]
    ks: torch.Tensor,
    brdf_lut: torch.Tensor,
):
    H, W, _ = normals.shape

    nrmvec = normals.reshape(1, H, W, 3)
    view_dirs = view_dirs.reshape(1, H, W, 3)
    diffuse_raw = kd.reshape(1, H, W, 3)
    roughness = kr.reshape(1, H, W, 1)
    spec_col = ks.reshape(1, H, W, 3)
    diff_col = 1.0 - spec_col

    reflvec = (
        2.0 * (nrmvec * view_dirs).sum(-1, keepdims=True).clamp(min=0.0) * nrmvec - view_dirs
    )  # [1, H, W, 3]

    ambient = dr.texture(light.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
    specular_linear = ambient * diff_col

    # spec0 = specular_linear.squeeze().permute(2,0,1)

    NoV = saturate_dot(nrmvec, view_dirs)  # [1, H, W, 1]
    fg_uv = torch.cat((NoV, roughness), dim=-1)  # [1, H, W, 2]
    fg_lookup = dr.texture(
        brdf_lut,  # [1, 256, 256, 2]
        fg_uv.contiguous(),  # [1, H, W, 2]
        filter_mode="linear",
        boundary_mode="clamp",
    ) 
    miplevel = light.get_mip(roughness)  # [1, H, W, 1]
    spec = dr.texture(light.specular[0][None, ...], 
                          reflvec.contiguous(), 
                          mip=list(m[None, ...] for m in light.specular[1:]), 
                          mip_level_bias=miplevel[..., 0], 
                          filter_mode='linear-mipmap-linear', 
                          boundary_mode='cube') # [1, H, W, 3]

    # mask = (nrmvec != 0).all(0, keepdim=True) #[1,h,w,3] 都不是0，才为true
    # zero_tensor = torch.zeros_like(reflvec).cuda()
    # reflvec_masked = torch.where(mask, reflvec, zero_tensor)
    # spec_masked = dr.texture(light.specular[0][None, ...], 
    #                          reflvec_masked.contiguous(), 
    #                          mip=list(m[None, ...] for m in light.specular[1:]),
    #                          mip_level_bias=miplevel[..., 0], 
    #                          filter_mode='linear-mipmap-linear',
    #                          boundary_mode='cube') 

    reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
    # spec1 = (spec * reflectance).squeeze().permute(2,0,1)
    
    
    specular_linear += spec * reflectance

    extras = {"specular_color": specular_linear.squeeze().permute(2,0,1)}

    # extras['spec0'] = spec0
    # extras['spec1'] = spec1

    diffuse_linear = torch.sigmoid(diffuse_raw - np.log(3.0))
    extras["diffuse_color"] = diffuse_linear.squeeze().permute(2,0,1)
    

    rgb = specular_linear + diffuse_linear

    return rgb.squeeze(), extras
    


