#!/bin/bash

#echo "python3 compress.py utk-face --pre predictive --ordering random --prev-context DABX --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear --mode single"

echo "python3 compress.py utk-face --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear --mode single"
python3 compress.py utk-face --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear --mode single
mv args.out utkface_2_mst_3000_imgs_dab_dab_linear_args.out
mv comp.out utkface_2_mst_3000_imgs_dab_dab_linear_comp.out
utk_mst_2_dab_dab_tp=$(ls -t true_pix* |  head -1)
utk_mst_2_dab_dab_tc=$(ls -t training_context* |  head -1)
echo "generated training files $utk_mst_2_dab_dab_tp and $utk_mst_2_dab_dab_tc"
mv $utk_mst_2_dab_dab_tp utk_mst_2_dab_dab_3000_tp.npy
mv $utk_mst_2_dab_dab_tc utk_mst_2_dab_dab_3000_tc.npy


#TODO echo "python3 compress.py utk-face --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear --mode single"

