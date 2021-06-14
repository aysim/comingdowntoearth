### Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization 
<img src="./teaser-small.png">
Implementation of the CVPR 2021 paper <a href="https://arxiv.org/pdf/2103.06818.pdf">Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization</a>. 

### Abstract 

The goal of cross-view image based geo-localization is to determine the location of a given street view image by matching it against a collection of geo-tagged satellite im- ages. This task is notoriously challenging due to the drastic viewpoint and appearance differences between the two do- mains. We show that we can address this discrepancy ex- plicitly by learning to synthesize realistic street views from satellite inputs. Following this observation, we propose a novel multi-task architecture in which image synthesis and retrieval are considered jointly. The rationale behind this is that we can bias our network to learn latent feature rep- resentations that are useful for retrieval if we utilize them to generate images across the two input domains. To the best of our knowledge, ours is the first approach that cre- ates realistic street views from satellite images and local- izes the corresponding query street-view simultaneously in an end-to-end manner. In our experiments, we obtain state- of-the-art performance on the CVUSA and CVACT bench- marks. Finally, we show compelling qualitative results for satellite-to-street view synthesis.

The code has been implemented & tested with Python 3.6.9 and Pytorch 1.5.0.
### Usage 
## Datasets and Polar transformation

* In our experiments, we use CVUSA and CVACT datasets. Use <a href="https://github.com/viibridges/crossnet">this link</a> to reach the CVUSA dataset and use <a href="https://github.com/Liumouliu/OriCNN">this link</a> to reach the CVACT dataset. 
* Change dataset paths under `helper/parser.py` and `helper/parser_cvact.py`. 
* The polar transformation script is copied from https://github.com/shiyujiao/cross_view_localization_SAFA/blob/master/script/data_preparation.py, you can find it `/data/convert_polar.py`.

## Pretrained Models 
* You can find the CVUSA pre-trained model under: https://vision.in.tum.de/webshare/u/toker/coming_dte_ckp/cvusa/.
* You can find the CVACT pre-trained model under: https://vision.in.tum.de/webshare/u/toker/coming_dte_ckp/cvact/.
## Train 
* For CVUSA, use `train_synthesis_cvusa.py` 
* For CVACT, use `train_synthesis_cvact.py` 

## Test 
To test our architecture use the pretained models, given above, and run  `cvusa_test.py` and `cvact_test.py`.
## Cite
If you use our implementation, please cite:
```
@article{toker2021coming,
  title={Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization},
  author={Toker, Aysim and Zhou, Qunjie and Maximov, Maxim and Leal-Taix{\'e}, Laura},
  journal={arXiv preprint arXiv:2103.06818},
  year={2021}
}
```
