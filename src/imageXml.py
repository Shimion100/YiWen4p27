"""
Created on 2016-09-13
Create a xml file
@author: gwd
"""
from xml.dom import minidom
def imageBuildXml(data):
    """
    this is used for build xml file to save to java.
    :param data:
    :return:
    """
    """
    document=minidom.Document()
    document.appendChild(document.createComment("this is used for save a image file"))
    imagelist=document.createElement("Images")
    document.appendChild(imagelist);
    """

def addImageNode(document=minidom.Document(),id=str(0),data=str(0)):
    image=document.createElement("Image")
    id=document.createElement("Id")
    data=document.createElement("Data")
    id.appendChild(document.createTextNode(id))
    data.appendChild(document.createTextNode(data))
    image.appendChild(id)
    image.appendChild(data)