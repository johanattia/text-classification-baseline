# baseline-text-classification
Deep Learning Baselines for Text Classification with TensorFlow

## Available Models
* Convolutional Neural Networks from [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf) and [A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf)
* Multi-components models aggregating (sub)word-level RNN, character-level CNN and TF-IDF+MLP architectures

## Next steps: Language Models fine-tuning
Some useful resources and frameworks to fine BERT-like Language Models:
* [On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines](https://arxiv.org/pdf/2006.04884.pdf)
* Hugging Face [Transformers](https://huggingface.co/docs/transformers/index)
* TensorFlow stack through [tf-models-official](https://github.com/tensorflow/models/tree/master/official) and [TensorFlow Hub](https://www.tensorflow.org/hub). For example, see the following tutorial: [Fine-tuning a BERT model](https://www.tensorflow.org/text/tutorials/fine_tune_bert?hl=en) 

## License
[MIT License](LICENSE)
