# "Forensic Similarity Graph" 
The following is a non-official implementation of the work "Exposing Fake Images with Forensic Similarity Graphs" by Owen Mayer and Matthew C. Stamm, Deparment of Electrical and Computer Engineering Drexel University - Philadelphia, PA, USA. Part of the code is based on the [forensic similarity](http://omayer.gitlab.io/forensicsimilarity/) code base.


## Prerequisites 
*  python 3
*  python packages:
    *  tensorflow 1.14.0
    *  numpy 1.16.4
    *  tqdm
    *  igraph 0.8.2
    
* optional recommended python packages:
    *  jupyter notebook (for working with example scripts)
    *  matplotlib (for loading images in the examples)
    *  cairocffi 1.1.0 (for generating plot examples)
    *  pillow (for loading JPEG images)
    *  scikit-learn 0.22.1 (for metrics calculation)

## Getting Started

Please see the [jupyter notebook examples](https://github.com/lucamaiano/forensic_similarity_graph/tree/main/notebook) to get started.

The "src" folder contains all the code that you need to run the project. To begin, create a "data" folder in the main directory of this project and paste your dataset. Then, you can run the project and change some basic configuration inside the "test.py" file.


## Cite the original paper
For more information regarding this work, please refear to the original paper.

bibtex:
```
@article{mayer2019forensicgraph,
  title={Exposing Fake Images with Forensic Similarity Graphs},
  author={Mayer, Owen and Stamm, Matthew C},
  journal={arXiv preprint arXiv:1912.02861},
  year={2019}
}
```
