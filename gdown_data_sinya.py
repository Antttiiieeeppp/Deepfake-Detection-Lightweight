import gdown

'''
#### List of zip folders ######
sample_dataset = https://drive.google.com/file/d/1-B818uKvxScwHn-lQo77ASL6MAuKz5Ot/
dataset_01 = https://drive.google.com/file/d/1bLvw54C2Rpv039zZhevuPX2MmnwKSsVw/
train_video data with json files = 'https://drive.google.com/file/d/1-CwFgB6HIa-3VVcAGPBcA5gDYLhD5T_k/'
test_video = https://drive.google.com/file/d/1-Eid6GGOeg19yQHDzdjNXCy89WM0o9LV/

Efficient_backbones = 'https://drive.google.com/drive/folders/1-x3Nh7rY_JHiVJTiZ6AikfiFM4rwPKox?dmr=1&ec=wgc-drive-globalnav-goto'
'''

#file_id = "1-Eid6GGOeg19yQHDzdjNXCy89WM0o9LV"
#url = f"https://drive.google.com/uc?id={file_id}"
url = f"https://drive.google.com/drive/folders/1-x3Nh7rY_JHiVJTiZ6AikfiFM4rwPKox?dmr=1&ec=wgc-drive-globalnav-goto"
output = "/home/work/Antttiiieeeppp/Efficient-AI-Backbones.zip"


gdown.download(url, output, quiet=False)
