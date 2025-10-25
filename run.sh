python kuwahara_filters.py --input 1.jpg --mode simple --radius 5 --output 1_out_simple.png
python kuwahara_filters.py --input 1.jpg --mode generalized --radius 6 --sectors 8 --q 8 --samples 6 --output 1_out_gen.png
python kuwahara_filters.py --input 1.jpg --mode anisotropic --radius 6 --sectors 8 --q 8 --samples 6 --aniso 2.0 --output 1_out_aniso.png

python kuwahara_filters.py --input 2.jpg --mode simple --radius 5 --output 2_out_simple.png
python kuwahara_filters.py --input 2.jpg --mode generalized --radius 6 --sectors 8 --q 8 --samples 6 --output 2_out_gen.png
python kuwahara_filters.py --input 2.jpg --mode anisotropic --radius 6 --sectors 8 --q 8 --samples 6 --aniso 2.0 --output 2_out_aniso.png

python kuwahara_filters.py --input 3.jpg --mode simple --radius 5 --output 3_out_simple.png
python kuwahara_filters.py --input 3.jpg --mode generalized --radius 6 --sectors 8 --q 8 --samples 6 --output 3_out_gen.png
python kuwahara_filters.py --input 3.jpg --mode anisotropic --radius 6 --sectors 8 --q 8 --samples 6 --aniso 2.0 --output 3_out_aniso.png
