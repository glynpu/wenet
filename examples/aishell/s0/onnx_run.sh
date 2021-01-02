source ~/env/py3/bin/activate
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../:$PYTHONPATH
stage=$1

onnx_model_path=converted_onnx_models/
mkdir $onnx_model_path
# onnx_model_path=preconverted_onnx_models
# model size(128M) exceeds GitHub's file size limit of 100.00 MB
# So I upload this model to baiducloud
# Download pretrained model from baiducloud to ./exp/
# Link: https://pan.baidu.com/s/1LL7ty4SxVGxpXWmKo-rtvQ 
# Code: bxzd
if [ ${stage} -le 1 ]; then
	python wenet/bin/export_onnx.py \
		--gpu 7 \
		--config exp/train.yaml \
		--test_data fbank_pitch/test/format.data \
		--checkpoint ./exp/final.pt \
		--beam_size 5 \
		--batch_size 1 \
		--penalty 0.0 \
		--dict data/dict/lang_char.txt \
		--cmvn fbank_pitch/train_sp/global_cmvn \
		--onnx_model_path ${onnx_model_path}

	# python change_dynamic_input_length.py
	python wenet/bin/change_dynamic_input_length.py \
		--onnx_model_path ${onnx_model_path}
fi
if [ ${stage} -eq 2 ]; then
	python wenet/bin/onnx_recognize.py \
		--gpu 7 \
		--config exp/train.yaml \
		--test_data fbank_pitch/test/format.data  \
		--beam_size 5 \
		--batch_size 1 \
		--dict data/dict/lang_char.txt \
		--cmvn fbank_pitch/train_sp/global_cmvn \
		--onnx_encoder_path=${onnx_model_path}/dynamic_dim_encoder.onnx \
		--onnx_decoder_init_path=${onnx_model_path}/dynamic_dim_decoder_init.onnx \
		--onnx_decoder_non_init_path=${onnx_model_path}/dynamic_dim_decoder_non_init.onnx
fi

