#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import argparse
import os
import time
import zipfile
import tarfile
import tempfile

import PIL.Image
import socket
from PIL import Image
import numpy as np
import scipy.misc
from google.protobuf import text_format
import threading
from Queue import Queue
import time
import random

import mylib

q = Queue(4)
lock = threading.Lock()

os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
import caffe
from caffe.proto import caffe_pb2

a = ["n02112706", "n02113023", "n02113186", "n02113624", "n02113712", "n02113799", "n02113978", "n02123045", "n02123159", "n02123394", "n02123597", "n02124075", "n02701002", "n02704792", "n02747177", "n02791124", "n02795169", "n02797295", "n02814533", "n02835271", "n02917067", "n02930766", "n02971356", "n02977058", "n03045698", "n03100240", "n03126707", "n03127925", "n03179701", "n03201208", "n03272562", "n03345487", "n03384352", "n03393912", "n03404251", "n03417042", "n03444034", "n03445924", "n03459775", "n03478589", "n03594734", "n03594945", "n03595614", "n03599486", "n03630383", "n03642806", "n03649909", "n03670208", "n03710637", "n03710721", "n03769881", "n03770439", "n03770679", "n03777568", "n03785016", "n03791053", "n03792782", "n03796401", "n03891251", "n03899768", "n03902125", "n03977966", "n04065272", "n04238763", "n04370456", "n04479046", "n04507155", "n04604644", "n06794110", "n06874185", "n07878787"]
b = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "3", "3", "3", "2", "2", "3", "2", "3", "3", "3", "1", "1", "1", "3", "3", "2", "1", "1", "2", "3", "3", "3", "1", "3", "2", "3", "1", "1", "1", "3", "1", "2", "1", "1", "3", "3", "1", "1", "3", "3", "3", "3", "3", "3", "3", "3", "1", "1", "1", "3", "3", "1", "1", "1", "1", "3", "3", "3", "3"]

net=0
transformer = 0
caffemodel = 0
deploy_file = 0
mean_file = 0
labels_file = 0
semaphore = 0

checkFlag=0

addr = 0
c = 0
s = 0
port = 5028

def unzip_archive(archive):
    tmpdir = os.path.join(tempfile.gettempdir(),
            os.path.basename(archive))
    assert tmpdir != archive # That wouldn't work out

    if os.path.exists(tmpdir):
        # files are already extracted
        pass
    else:
        if tarfile.is_tarfile(archive):
            print 'Extracting tarfile ...'
            with tarfile.open(archive) as tf:
                tf.extractall(path=tmpdir)
        elif zipfile.is_zipfile(archive):
            print 'Extracting zipfile ...'
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(path=tmpdir)
        else:
            raise ValueError('Unknown file type for %s' % os.path.basename(archive))
    return tmpdir

#############################################################################################
def get_net(caffemodel, deploy_file, use_gpu=True):

    if use_gpu:
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)

def get_transformer(deploy_file, mean_file=None):
	
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file,'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

def load_image(path, height, width, mode='RGB'):
	
    #print '????: %s' % path
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

def forward_pass(images, batch_size=1):
	
    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)

    caffe_images = np.array(caffe_images)

    dims = transformer.inputs['data'][1:]
    
    
	
    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        output = net.forward()[net.outputs[-1]]
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        print 'Processed %s/%s images ...' % (len(scores), len(caffe_images))
        
    

    return scores

def read_labels(labels_file):
	
    if not labels_file:
        print 'WARNING: No labels file provided. Results will be difficult to interpret.'
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels

    

###################################after Here need to Thread############################
#def classify( caffemodel, deploy_file,image_files, mean_file, labels_file, use_gpu, net, transformer):
def classify(image_files, use_gpu):
	
    _, channels, height, width = transformer.inputs['data']
    
    startTime = time.time()
    
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
        
        
    print '%s' % image_files
    imagetime = time.time()
    images = [load_image(image_file, height, width, mode) for image_file in image_files]

    labeltime = time.time()
    labels = read_labels(labels_file)

    # Classify the image
    classify_start_time = time.time()
    scores = forward_pass(images)
    print 'Classification took %s seconds.' % (time.time() - classify_start_time,)
    #print scores

    ### Process the results

    indices = (-scores).argsort()[:, :5] # take top 5 results
    classifications = []
    for image_index, index_list in enumerate(indices):
        result = []
        for i in index_list:
            # 'i' is a category in labels and also an index into scores
            if labels is None:
                label = 'Class #%s' % i
            else:
                label = labels[i]
            result.append((label, round(100.0*scores[image_index, i],4)))
        classifications.append(result)

    print
    print

    tempbbb = time.time()
    idx = 0
    predicArr = [[0 for col in range(10)] for row in range(10)]

    for index, classification in enumerate(classifications):
        print '{:-^80}'.format(' Prediction for %s ' % image_files[index])
        for label, confidence in classification:
            print '{:9.4%} - "{}"'.format(confidence/100.0, label)
            predicArr[idx][0] = confidence/100.0
            predicArr[idx][1] = label      
            idx=idx+1
        print

    print 'Total prediction Time %s' % (time.time() - startTime)
    print
    
    return predicArr

#############################################################################################
    

def preprocessFunc(archive, use_gpu=True):
	
    startTime = time.time()

    tmpdir = unzip_archive(archive)
    global caffemodel 
    caffemodel = None
    global deploy_file 
    deploy_file = None
    global mean_file 
    mean_file = None
    global labels_file 
    labels_file = None
    
    for filename in os.listdir(tmpdir):
        full_path = os.path.join(tmpdir, filename)
        if filename.endswith('.caffemodel'):
            caffemodel = full_path
        elif filename == 'deploy.prototxt':
            deploy_file = full_path
        elif filename.endswith('.binaryproto'):
            mean_file = full_path
        elif filename == 'labels.txt':
            labels_file = full_path
        else:
            print 'Unknown file:', filename

    assert caffemodel is not None, 'Caffe model file not found'
    assert deploy_file is not None, 'Deploy file not found'

    
    startTime = time.time()

    # Load the model and images
    global net
    net = get_net(caffemodel, deploy_file, True) 
    
    print 'get_net lead time (read model instance) : %s' % (time.time() - startTime)

    get_net_time = time.time()

    global transformer 
    transformer = get_transformer(deploy_file, mean_file)

    print 'after transfomer(get tranform) : %s' %(time.time() - get_net_time)


def checkImage():
	
	k = 0
	global semaphore
	global s
	global c
	global addr
	global port
	global checkFlag
	

	while True :
		
		print '이미지를 받을 준비가 되었음!'
		semaphore = 0

		k = k+1
		buf = ['/home/ubuntu/capston/transImage/image%d.jpg'%(k)]
		buf1 = '/home/ubuntu/capston/transImage/image%d.jpg'%(k)
	
		#if(checkFlag == 0) :
			
		s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
		
		s.bind(('', port))
		

		
		print "AAA"
		f = open(buf1, 'wb')
		s.listen(2)

		c, addr = s.accept()

		startTime = time.time()
	
		print 'Got connection from', addr
		print "Receiving..."
		l = c.recv(1024)
		qqq=0
		while (1):
			qqq = qqq + 1
			#print "Receiving.. %s" % qqq
			#print "%s"%l
			f.write(l)
			
			if len(l) != 1024 :
				print "%s"%qqq
				break
				
			l = c.recv(1024)
			
		f.close()
		print "Done Receving"

		print 'send : %s -> Total Time'%buf,time.time() - startTime
		
		#c.send('Thank you for connecting')
		#c.close()
		
		
		
		global lock
		lock.acquire()
			
		if q.full() == True :
			k=k-1
			lock.release()
			continue
		
		q.put(buf)
		
		lock.release()
	
		while semaphore == 0: # busy waiting
			no = 3
			
		#time.sleep(1)
	
def classification():
	
	global semaphore
	global s
	global c
	global addr
	global port
	
	############################preprocess############################### 이 함수는 한번만 실행되면 됨.
	preprocessFunc(args['archive'], True)
	############################preprocess############################### 이 함수는 한번만 실행되면 됨.
	
	
	while True : 
		
		st = time.time()
		
		global lock
		
		lock.acquire()
		
		if q.empty() == True :
			lock.release()
			continue
	
		answerArr = [[0 for col in range(6)] for row in range(6)]
		
		inputImage = q.get()
		
		#answerArr = classify(caffemodel, deploy_file, inputImage ,mean_file, labels_file, True,net,transformer)
		answerArr = classify(inputImage , True)
		
		lock.release()
		
		for i in range(0,1):
			print '%s %s' %(answerArr[i][0]*100 , answerArr[i][1])
			resStr  = answerArr[i][1][0]
			resStr += answerArr[i][1][1]
			resStr += answerArr[i][1][2]
			resStr += answerArr[i][1][3]
			resStr += answerArr[i][1][4]
			resStr += answerArr[i][1][5]
			resStr += answerArr[i][1][6]
			resStr += answerArr[i][1][7]
			resStr += answerArr[i][1][8]
			
			print 'ALLTIME : %s'%(time.time() - st)
			print 'result : %s'%(resStr)
			
			################################################3
			#port  = 5003

			#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			#s.bind(('', port))
			#s.listen(1)

			#c, addr = s.accept()

			print 'Connected by', addr
			data = resStr
			for x in range(len(a)):
				if data == a[x]:
					print "aaaaaaaaaaaaaaaaaaaaaaaa %s"%a[x]
					print "bbbbbbbbbbbbbbbbbbbbbbbb %s"%b[x]
					c.send(b[x])
	
					c.close()
					s.close()
					break
			###################################################
			
			os.system('play -q ../sound/front.mp3')
			
			mylib.wlog(resStr)
			
			print '이미지 넘버 -> %s' % resStr
			print '이미지 경로 -> %s' % inputImage
			
			os.system('play -q ../sound/isthere.mp3')
			
			print '*********************************************************************'
			
			
			
			semaphore = 1
	

if __name__ == '__main__':
    script_start_time = time.time()
    
    

    parser = argparse.ArgumentParser(description='Classification example using an archive - DIGITS')

    ### Positional arguments

    parser.add_argument('archive',  help='Path to a DIGITS model archive')

    parser.add_argument('--nogpu',action='store_true',help="Don't use the GPU")

    args = vars(parser.parse_args())

	############################preprocess############################### 이 함수는 한번만 실행되면 됨.
    
    #preprocessFunc(args['archive'], True)
    
    #####################################################################
    
    ##################################################################### 이미지가 추가되었는지 판단하는 스레드
    t1 = threading.Thread(target=checkImage);
    t1.start()
    #####################################################################


	##################################################################### 이미지 식별하는 스레드
    t2 = threading.Thread(target=classification);
    t2.start()
    #####################################################################    
    startTime = time.time()
    

