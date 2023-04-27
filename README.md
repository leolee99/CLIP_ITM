# CLIP for Image-text Retrieval

A simple pytorch implementation of baseline based-on [**CLIP**](https://arxiv.org/abs/2103.00020) for Image-text Retrieval.

This project provides a CLIP-based training and evaluation framework for image-text retrieval on commonly used MS-COCO dataset.


## Requirements
We recommend the following dependencies.

* Python 3.8
* [PyTorch](http://pytorch.org/) 1.7.1
* [NumPy](http://www.numpy.org/) (>1.19.5)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

```bash
pip install requirments.txt
```

## Dataset Preparation

### COCO Caption

We follow the same split provided by [VSE++](http://www.cs.toronto.edu/~faghri/vsepp/data.tar).
Dataset splits can be found in [datasets/annotations](datasets/annotations).

The final data directory tree should be:
```
${DATAPATH}/
├── annotations/
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   ├── coco_train_ids.npy
│   ├── coco_dev_ids.npy
│   ├── coco_test_ids.npy
│   ├──coco_restval_ids.npy
│   └── ...
│          
└── images/ # all images of MS-COCO
```

## Training

You can finetune the model by running:

**ViT-B/32:**
```bash
python main.py --batch_size 256 --epochs 10 --lr 1e-5 --warmup 500 --vision_model ViT-B/32 --dataset_root ${DATAPATH}
```

**ViT-B/16:**
```bash
python main.py --batch_size 128 --epochs 5 --lr 1e-5 --warmup 500 --vision_model ViT-B/16
```

## Evaluation

You can eval the model by running:
```bash
python main.py --eval --resume ${MODELPATH} --vision_model ${VISONMODEL}
```

## Zero-shot Results on MS-COCO

<table>
    <tr>
        <td></td>
        <td colspan="3"><center><b>Image-to-text</b></center></td>
        <td colspan="3"><center><b>Text-to-image</b></center></td>
    </tr>
    <tr>
        <td><b>Vision model</b></td>
        <td><b>R@1</b></td>
        <td><b>R@5</b></td>
        <td><b>R@10</b></td>
        <td><b>R@1</b></td>
        <td><b>R@5</b></td>
        <td><b>R@10</b></td>
    </tr>
    <tr>
        <td>RN50</td>
        <td>49.10</td>
        <td>73.04</td>
        <td>82.02</td>
        <td>28.56</td>
        <td>53.00</td>
        <td>64.54</td>
    </tr>
    <tr>
        <td>RN50x4</td>
        <td>53.12</td>
        <td>76.90</td>
        <td>84.82</td>
        <td>33.42</td>
        <td>58.10</td>
        <td>68.36</td>
    </tr>
    <tr>
        <td>RN50x16</td>
        <td>55.24</td>
        <td>78.68</td>
        <td>86.60</td>
        <td>35.45</td>
        <td>60.05</td>
        <td>70.12</td>
    </tr>
    <tr>
        <td>RN50x64</td>
        <td>58.60</td>
        <td>80.70</td>
        <td>87.60</td>
        <td>35.45</td>
        <td>59.92</td>
        <td>70.20</td>
    </tr>
    <tr>
        <td>RN101</td>
        <td>49.56</td>
        <td>74.48</td>
        <td>82.38</td>
        <td>30.65</td>
        <td>55.47</td>
        <td>66.06</td>
    </tr>
    <tr>
        <td>ViT-B/32</td>
        <td>50.16</td>
        <td>75.02</td>
        <td>83.58</td>
        <td>30.42</td>
        <td>56.04</td>
        <td>66.88</td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>52.38</td>
        <td>76.86</td>
        <td>84.76</td>
        <td>33.05</td>
        <td>58.49</td>
        <td>69.16</td>
    </tr>
    <tr>
        <td>ViT-L/14</td>
        <td>56.36</td>
        <td>79.50</td>
        <td>86.66</td>
        <td>36.54</td>
        <td>60.97</td>
        <td>71.16</td>
    </tr>
    <tr>
        <td>ViT-L/14 (336px)</td>
        <td>58.06</td>
        <td>81.12</td>
        <td>87.92</td>
        <td>37.18</td>
        <td>61.59</td>
        <td>71.42</td>
    </tr>
</table>

## Fine-tuned Results on MS-COCO 5K
<table>
    <tr>
        <td></td>
        <td colspan="3"><center><b>Image-to-text</b></center></td>
        <td colspan="3"><center><b>Text-to-image</b></center></td>
    </tr>
    <tr>
        <td><b>Vision model</b></td>
        <td><b>R@1</b></td>
        <td><b>R@5</b></td>
        <td><b>R@10</b></td>
        <td><b>R@1</b></td>
        <td><b>R@5</b></td>
        <td><b>R@10</b></td>
    </tr>
    <tr>
        <td>ViT-32/B</td>
        <td>62.22</td>
        <td>85.62</td>
        <td>91.66</td>
        <td>46.94</td>
        <td>74.88</td>
        <td>83.56</td>
    </tr>
    <tr>
        <td>ViT-16/B</td>
        <td>68.76</td>
        <td>88.66</td>
        <td>93.94</td>
        <td>52.45</td>
        <td>78.66</td>
        <td>86.66</td>
    </tr>

## Planning

Providing the training and evaluation codes on Flickr30K.