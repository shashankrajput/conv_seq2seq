export PYTHONIOENCODING=UTF-8

export VOCAB_SOURCE=${HOME}/Workspace/data/nmt_data/wmt16_en_de/vocab.bpe.32000
export VOCAB_TARGET=${HOME}/Workspace/data/nmt_data/wmt16_en_de/vocab.bpe.32000
export TRAIN_SOURCES=${HOME}/Workspace/data/nmt_data/wmt16_en_de/train.tok.clean.bpe.32000.en
export TRAIN_TARGETS=${HOME}/Workspace/data/nmt_data/wmt16_en_de/train.tok.clean.bpe.32000.de
export DEV_SOURCES=${HOME}/Workspace/data/nmt_data/wmt16_en_de/newstest2013.tok.bpe.32000.en
export DEV_TARGETS=${HOME}/Workspace/data/nmt_data/wmt16_en_de/newstest2013.tok.bpe.32000.de

export DEV_TARGETS_REF=${HOME}/Workspace/data/nmt_data/wmt16_en_de/newstest2013.tok.de
export TRAIN_STEPS=1000000

export MODEL_DIR=${HOME}/Workspace/checkpoints/nmt_models/nmt_conv_seq2seq_prob
mkdir -p $MODEL_DIR


python -m bin.train \
  --config_paths="
      ./example_configs/conv_prob_seq2seq.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipelineFairseq
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --eval_every_n_steps 50 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR



'''
export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

###with greedy search
python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 1 
    decoder.class: seq2seq.decoders.ConvDecoderFairseq" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/predictions.txt

'''

'''
###with beam search
python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5 
    decoder.class: seq2seq.decoders.ConvDecoderFairseqBS" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/predictions.txt


./bin/tools/multi-bleu.perl ${TEST_TARGETS} < ${PRED_DIR}/predictions.txt
'''



