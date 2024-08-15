branch main,gaushader都是--先pergaussian shading, 再blend.


main是用的gsir的参数化方式，gaushader是用的GaussianShader的参数化方式.


branch equivariant里面pbr_deffered_train/render 是先blend，再deffered shading，用的是GaussianShader的参数化方式.