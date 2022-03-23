#!/bin/bash
python preprocessing_sensordata.py --dir_path ./raw_test/ --save_path save_test_torch/ --num_files 1000
python load_pretrain_model.py --src_path save_test_torch