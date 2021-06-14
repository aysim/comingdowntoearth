<h1> Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization </h1>
<img src="./teaser-small.png">
Implementation of the CVPR 2021 paper <a href="https://arxiv.org/pdf/2103.06818.pdf">Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization</a>. 
<h2> Dependencies </h2>
The code has been implemented & tested with Python 3.6.9 and Pytorch 1.5.0.
<h2> Usage </h2>
<h3> Datasets and Polar transformation</h3>
<ul>
<li> In our experiments, we use CVUSA and CVACT datasets. Use <a href="https://github.com/viibridges/crossnet">this link</a> to reach the CVUSA dataset and use <a href="https://github.com/Liumouliu/OriCNN">this link</a> to reach the CVACT dataset. </li>
<li> Change dataset paths under helper/parser.py and helper/parser_cvact.py. </li>
<li> The polar transformation script is copied from https://github.com/shiyujiao/cross_view_localization_SAFA/blob/master/script/data_preparation.py, you can find it /data/convert_polar.py </li>
<h3> Pretrained Models </h3>
