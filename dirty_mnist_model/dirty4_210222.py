from PIL import Image
from pytesseract import *
import configparser
import os

config = configparser.ConfigParser()
config.read(os.path.dirname(os.path.realpath(__file__))
+os.sep+'envs'+os.sep+'property.ini')

def ocrToStr(fullpath, outTxtpath,fileName, lang='eng'):
    img = Image.open(fullpath)
    txtName = os.path.join(outTxtpath,fileName.split('.')[0])
    outText = image_to_string(img, lang=lang, 
    config = '--psm 1 -c preserve_interword_spaces=1')
    print('++OCT Extract Result +++')
    print('Extract FileName-->> :', fileName, ' : <<--')
    print('\n\n')
    print(outText)
    strToTxt(txtName, outText)

def strToTxt(txtName, outText):
    with open(txtName + '.txt', 'w', encoding = 'utf-8') as f:
        f.write(outText)

if __name__ == "__main__":
    outTxtPath = os.path.dirname(os.path.realpath(__file__)) + config['Path']['OcrTxtPath']
    for root, dirs, files in os.walk(os.path.dirname(os.path.relpath(__file__)) + config['Path']['OriImgPath']):
        for fname in files:
            fullName = os.path.join(root, fname)
            ocrToStr(fullName, outTxtPath, fname, 'kor+eng')