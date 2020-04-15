'''
Sometimes we want to encrypt our data... 2020/4/9
'''

#coding:utf-8
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad,unpad
from base64 import b64encode,b64decode
from random import choice
import json
import os.path
import numpy as np
import cv2

class AES_CBC:
    __S=r'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/=@#$%^&*'
    
    def encrypt(self,data,key=None,file=None):
        '''encrypt data with key(if None, generate randomly), if you give arg file, we will 
           save iv(a generated initialization vector) and ct(the cipher text) into local. 
           Note that key's length must be 16 or 32. Func will return the key. See doc in 
           https://pycryptodome.readthedocs.io/en/latest/src/cipher/classic.html#cbc-mode'''
        if key==None or (len(key) not in [16,32]):
            if key!=None:
                print('WARNNING: key\'s length must be 16 or 32!')
            key=''.join([choice(self.__S) for i in range(choice([16,32]))])
            print('KEY: %s'%key)
        key=self.to_bytes(key)
        cipher=AES.new(key,AES.MODE_CBC)
        ct_bytes=cipher.encrypt(pad(self.to_bytes(data),AES.block_size))
        iv=b64encode(cipher.iv).decode('utf-8')
        ct=b64encode(ct_bytes).decode('utf-8')
        self.iv_ct={'iv':iv, 'ct':ct}
        if file!=None:
            with open(file,'w') as f:
                json.dump(self.iv_ct,f)
        return key.decode('utf-8')
    
    def decrypt(self,key,iv_ct=None):
        '''decrypt iv_ct with key, iv_ct can be a json file string(to load iv and ct), a dict
           (contains iv and ct), otherwise we will try to use self.iv_ct, note that self.iv_ct
            always records the latest one we visited. Func will return the decrypted data'''
        if isinstance(iv_ct,str) and os.path.exists(iv_ct):
            with open(iv_ct,'r') as f:
                self.iv_ct=json.load(f)
        elif isinstance(iv_ct,dict) and (iv_ct.get('iv')!=None and iv_ct.get('ct')!=None):
            self.iv_ct=iv_ct
        elif self.__dict__.get('iv_ct')==None:
            raise ValueError('must give the info of iv and ct!')
        iv=b64decode(self.iv_ct['iv'])
        ct=b64decode(self.iv_ct['ct'])
        key=self.to_bytes(key)
        try:
            cipher=AES.new(key,AES.MODE_CBC,iv)
            pt=unpad(cipher.decrypt(ct),AES.block_size)
        except:
            print('ERROR: Incorrect AES key!')
            return None
        return pt.decode('utf-8')
    
    @staticmethod
    def to_bytes(data):
        return data if isinstance(data,bytes) else bytes(data,encoding='utf-8')

class AES_CBC_IMG(AES_CBC):
    '''encrypt image(ndarray object) and decrypt. Note that it always makes encrypted data very 
       very large'''
    def encrypt(self,img,key=None,file=None):
        if isinstance(img,str) and os.path.exists(img):
            img=cv2.imread(img)
        elif not isinstance(img,np.ndarray):
            raise ValueError('arg img must be ndarray or a path of image!')
        json_img=json.dumps(img.tolist())
        return super().encrypt(json_img,key,file)
    
    def decrypt(self,key,iv_ct=None):
        return np.array(json.loads(super().decrypt(key,iv_ct)))

if __name__ == "__main__":
    key = "1"*8
    aes=AES_CBC()
    key=aes.encrypt('daiyang',key)
    data=aes.decrypt(key)
    print(data)