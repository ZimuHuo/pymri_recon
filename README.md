# pymri_recon

#I have much more features to come, this is just for a quick demo for the PhD application

MRI reconstruction package


## set up
Clone the github repo in to local 
```bash
git clone https://github.com/ZimuHuo/pymri_recon.git
```
Create conda env from .yml files 
```bash
cd pymri_recon
conda env create -f environment.yml
```
Acvtivate the conda environment
```bash
 conda activate pymrirecon
```
I used twixtools to read the file, you can also use pymapvbvd. 
```bash
git clone https://github.com/pehses/twixtools.git
cd twixtools
pip install .
```

Navigate to the pymrirecon folder and download example data
```bash
mkdir lib
cd lib
```

I prepared three example data. However, the dicoms are not available because my personal information is there.
https://drive.google.com/drive/folders/1-e5ywHM5BFP9od-_Yo_9E8E_1aYidrIy?usp=sharing

Example data: 
1. FLASH single slice phantom data 
2. 4-slice phantom data
3. Single-shot EPI in-vivo scan mid short axis view of left ventricle
