#!/bin/bash

#echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DABC --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear --feature-file mst_2_dabc_dab_tc.npy --label-file mst_2_dabc_dab_tp.npy"

#echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DABC --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 10 --predictor-family linear --feature-file mst_10_dabc_dab_tc.npy --label-file mst_10_dabc_dab_tp.npy"

#echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DABC --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family logistic --feature-file mst_2_dabc_dab_tc.npy --label-file mst_2_dabc_dab_tp.npy"

#echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family logistic --feature-file mst_2_dab_dab_tc.npy --label-file mst_2_dab_dab_tp.npy"

#echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DABC --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 10 --predictor-family logistic --feature-file mst_10_dabc_dab_tc.npy --label-file mst_10_dabc_dab_tp.npy"

#echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 10 --predictor-family linear"

#echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear --feature-file mst_2_dab_dab_tc.npy --label-file mst_2_dab_dab_tp.npy"

#echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DABC --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 10 --predictor-family logistic --feature-file mst_10_dabc_dab_tc.npy --label-file mst_10_dabc_dab_tp.npy"

#echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DABX --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear"

#echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DABX --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 10 --predictor-family linear"

#echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DABX --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 10 --predictor-family logistic"

#echo "python3 compress.py cifar-10 --pre predictive --ordering random --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear  --feature-file cifar_2_random_dab_dab_tc.npy --label-file cifar_2_random_dab_dab_tp.npy"

#echo "python3 compress.py cifar-10 --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear"

#echo "python3 compress.py cifar-10 --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 5 --predictor-family linear"

#echo "python3 compress.py cifar-10 --pre predictive --ordering mst --prev-context DABX --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear"

#echo "python3 compress.py cifar-10 --pre predictive --ordering mst --prev-context DABC --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear"

#echo "python3 compress.py cifar-10 --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear --mode triple --feature-file cifar_2_mst_dab_dab_triple_tc.npy --label-file cifar_2_mst_dab_dab_triple_tp.npy"

#echo "python3 compress.py cifar-10 --pre predictive --ordering mst --prev-context DABX --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear --mode triple --feature-file cifar_2_mst_dab_dab_triple_tc.npy --label-file cifar_2_mst_dab_dab_triple_tp.npy"

#echo "python3 compress.py cifar-10 --pre predictive --ordering mst --prev-context DABX --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear --feature-file cifar_2_mst_dabx_dab_tc.npy --label-file cifar_2_mst_dabx_dab_tp.npy --k 5"

#echo "python3 compress.py cifar-10 --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear --feature-file cifar_2_mst_dab_dab_tc.npy --label-file cifar_2_mst_dab_dab_tp.npy --k 5"

#echo "python3 compress.py cifar-10 --pre predictive --ordering mst --prev-context DABX --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear --feature-file cifar_2_mst_dabx_dab_tc.npy --label-file cifar_2_mst_dabx_dab_tp.npy --k 11"

#echo "python3 compress.py cifar-10 --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family linear --feature-file cifar_2_mst_dab_dab_tc.npy --label-file cifar_2_mst_dab_dab_tp.npy --k 11"

echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DABC --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family cubist"
python3 compress.py mnist --pre predictive --ordering mst --prev-context DABC --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family cubist
mv args.out mnist_2_mst_dabc_dab_cubist_args.out
mv comp.out mnist_2_mst_dabc_dab_cubist_comp.out

echo "python3 compress.py mnist --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family cubist"
python3 compress.py mnist --pre predictive --ordering mst --prev-context DAB --current-context DAB --comp predictive --enc pred-huff --num-prev-imgs 2 --predictor-family cubist
mv args.out mnist_2_mst_dab_dab_cubist_args.out
mv comp.out mnist_2_mst_dab_dab_cubist_comp.out
