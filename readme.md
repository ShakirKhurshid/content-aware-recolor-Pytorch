
# Content Aware Image Recoloarization
A deep neural network  which recolors an image according to a given tar-
get color that is useful to express image in vari-ous color concepts. The network is capable of
performing a content aware recolorization based on the target palette. 

## Usage
Run DL2022/run.py as

`python run.py -i 'path_to_input_image' -p 'hexcodes_of_color_palette' -m 'path_to_saved_model' -o 'path_of_output'`

example

`python run.py -i 'Samples/3/3.png' -p '#507b71' '#6caebc' '#6ead9c' '#afd9c3' '#b8dfdc' '#ecebd7' -m 'saved_model/model.pth' -o 'output.jpg'
