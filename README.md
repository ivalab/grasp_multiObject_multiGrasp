# grasp_multiObject_multiGrasp

This is the implementation of our RA-L work 'Real-world Multi-object, Multi-grasp Detection'. The detector takes RGB-D image input and predicts multiple grasp candidates for a single object or multiple objects, in a single shot. The original arxiv paper can be found [here](https://arxiv.org/pdf/1802.00520.pdf). The final version will be updated after publication process.

If you find it helpful for your research, please consider citing:

    @inproceedings{chu2018deep,
      title = {Real-World Multiobject, Multigrasp Detection},
      author = {F. Chu and R. Xu and P. A. Vela},
      journal = {IEEE Robotics and Automation Letters},
      year = {2018},
      volume = {3},
      number = {4},
      pages = {3355-3362},
      DOI = {10.1109/LRA.2018.2852777},
      ISSN = {2377-3766},
      month = {Oct}
    }


If you encounter any questions, please contact me at fujenchu[at]gatech[dot]edu


### Demo
1. Clone this repository
```
git clone https://github.com/ivalab/grasp_multiObject_multiGrasp.git
cd grasp_multiObject_multiGrasp
```

2. Build Cython modules
```
cd lib
make clean
make
cd ..
```

3. Install [Python COCO API](https://github.com/cocodataset/cocoapi)
```
cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..
```

4. Download pretrained models
- trained model for grasp on [dropbox drive](https://www.dropbox.com/s/ldapcpanzqdu7tc/models.zip?dl=0) 
- put under `output/res50/train/default/`

5. Run demo
```
./tools/demo_graspRGD.py --net res50 --dataset grasp
```
you can see images pop out.

### Train
1. Generate data   
1-1. Download [Cornell Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php)   
1-2. Run `dataPreprocessingTest_fasterrcnn_split.m` (please modify paths according to your structure)   
1-3. Follow 'Format Your Dataset' section [here](https://github.com/zeyuanxy/fast-rcnn/tree/master/help/train) to check if your data follows VOC format   

2. Train
```
./experiments/scripts/train_faster_rcnn.sh 0 graspRGB res50
```

### ROS version?
Yes! please find it [HERE](https://github.com/ivaROS/ros_deep_grasp)

### Acknowledgment

This repo borrows tons of code from
- [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) by endernewton

### Resources
- [multi-object grasp dataset](https://github.com/ivalab/grasp_multiObject)
- [grasp annotation tool](https://github.com/ivalab/grasp_annotation_tool)
