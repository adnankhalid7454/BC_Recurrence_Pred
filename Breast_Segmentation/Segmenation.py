import numpy as np
import torch
from scipy import ndimage as nd  
import torch.nn.parallel
from torch.autograd import Variable

def Generator_multichannels(image, sizeofchunk, sizeofchunk_expand,numofchannels): 
    sizeofimage = np.shape(image)[1:4]
            
    nb_chunks = (np.ceil(np.array(sizeofimage)/float(sizeofchunk))).astype(int)
    
    pad_image = np.zeros(([numofchannels,nb_chunks[0]*sizeofchunk,nb_chunks[1]*sizeofchunk,nb_chunks[2]*sizeofchunk]), dtype='float32')
    pad_image[:,:sizeofimage[0], :sizeofimage[1], :sizeofimage[2]] = image
            
    width = int(np.ceil((sizeofchunk_expand-sizeofchunk)/2.0))
            
    size_pad_im = np.shape(pad_image)[1:4]
    size_expand_im = np.array(size_pad_im) + 2 * width
    expand_image = np.zeros(([numofchannels,size_expand_im[0],size_expand_im[1],size_expand_im[2]]), dtype='float32')
    expand_image[:,width:-width, width:-width, width:-width] = pad_image
  
            
    batchsize = np.prod(nb_chunks)
    idx_chunk = 0
    chunk_batch = np.zeros((batchsize,numofchannels,sizeofchunk_expand,sizeofchunk_expand,sizeofchunk_expand),dtype='float32')
    idx_xyz = np.zeros((batchsize,3),dtype='uint16')
    for x_idx in range(nb_chunks[0]):
        for y_idx in range(nb_chunks[1]):
            for z_idx in range(nb_chunks[2]):
                
                idx_xyz[idx_chunk,:] = [x_idx,y_idx,z_idx]
                        
                         

                chunk_batch[idx_chunk,:,...] = expand_image[:,x_idx*sizeofchunk:x_idx*sizeofchunk+sizeofchunk_expand,\
                           y_idx*sizeofchunk:y_idx*sizeofchunk+sizeofchunk_expand,\
                           z_idx*sizeofchunk:z_idx*sizeofchunk+sizeofchunk_expand]             
                        
                idx_chunk += 1
    
    return chunk_batch, nb_chunks, idx_xyz, sizeofimage    

def Chunks_Image(segment_chunks, nb_chunks, sizeofchunk, sizeofchunk_expand, idx_xyz, sizeofimage):
    
    batchsize = np.size(segment_chunks,0)

    segment_image = np.zeros((nb_chunks[0]*sizeofchunk,nb_chunks[1]*sizeofchunk,nb_chunks[2]*sizeofchunk))
    
    for idx_chunk in range(batchsize):
        
        idx_low = idx_xyz[idx_chunk,:] * sizeofchunk
        idx_upp = (idx_xyz[idx_chunk,:]+1) * sizeofchunk
        
        segment_image[idx_low[0]:idx_upp[0],idx_low[1]:idx_upp[1],idx_low[2]:idx_upp[2]] = \
        segment_chunks[idx_chunk,0,...]
        

    segment_image = segment_image[:sizeofimage[0], :sizeofimage[1], :sizeofimage[2]]
    return segment_image



def BreastSeg(image,scale_subject,model,opt):
    modelpath = "Models/"       
    modelname = modelpath+"model_breast.pth"  
    checkpoint = torch.load(modelname)
    model.load_state_dict(checkpoint)
    numofseg =1
    commonspacing = [1.5,1.5,1.5]
   # commonspacing = [1.0,1.0,1.0]
    imageshape = image.shape

    scale_subject = scale_subject[::-1]/commonspacing    
    image = nd.interpolation.zoom(image,scale_subject,order=1)    
    imagesize = image.shape
    sizeofchunk = 20
    sizeofchunk_expand = 108    
    if opt.cuda:
        # sizeofchunk = 132
        # sizeofchunk_expand = 220   
         sizeofchunk = 20
         sizeofchunk_expand = 108  

    image_one = np.zeros((1,imagesize[0],imagesize[1],imagesize[2]),dtype='float32')
    image_one[0,...] =image 
    chunk_batch, nb_chunks, idx_xyz, sizeofimage = Generator_multichannels(image_one,sizeofchunk,sizeofchunk_expand,1)
    seg_batch = np.zeros((np.size(chunk_batch,0),numofseg,sizeofchunk,sizeofchunk,sizeofchunk),dtype='float32')
    for i_chunk in range(np.size(chunk_batch,0)):
        input = Variable(torch.from_numpy(chunk_batch[i_chunk:i_chunk+1,...]),volatile=True)
        model.eval()
        if opt.cuda:
            input = input.cuda()
        prediction = model(input)   
        
        seg_batch[i_chunk,0,...] = (prediction.data).cpu().numpy()
    for i_seg in range(numofseg):
        prob_image = Chunks_Image(seg_batch[:,i_seg:i_seg+1,...], nb_chunks, sizeofchunk, sizeofchunk_expand, idx_xyz, sizeofimage)
        up_image = nd.interpolation.zoom(prob_image,1/scale_subject,order=1)
        up_image_norm = np.zeros(imageshape,dtype='float32')
        temp_image = up_image[0:imageshape[0],0:imageshape[1],0:imageshape[2]]
        shape_tempimage =np.shape(temp_image)
        up_image_norm[0:shape_tempimage[0],0:shape_tempimage[1],0:shape_tempimage[2]] = temp_image
        
        threshold = 0.5
        idx = up_image_norm > threshold
        up_image_norm[idx] = 1
        up_image_norm[~idx] = 0               
        seg_img = up_image_norm.astype('uint8')
    return seg_img


