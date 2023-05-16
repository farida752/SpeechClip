import torch
import avssl.model
from avssl.module import ClipModel
import librosa
import os
import pickle
import math
import numpy as np
import gc
import sys
import time
# from pympler import asizeof


# batch size for speech inference

#embedding_size = 512
# start = 6414 + 1
start = 0

# load model to GPU
device = torch.device("cuda")

#use Parallel SpeechCLIP trained on Flickr8k for example
model_fp = "slt_ckpts/SpeechCLIP/large/flickr/parallel/epoch_56-step_6668-val_recall_mean_10_89.0000.ckpt"
model = avssl.model.KWClip_GeneralTransformer.load_from_checkpoint(model_fp)
model.to(device)
model.eval()

print("speech clid model loaded successfully")
# image model
model_img = ClipModel(
        "ViT-L/14",
        device=device,
        image_encoder_trainable=False,
        text_encoder_trainable=False,
    ).to(device)
model_img.eval()

# load input wav (should be 16kHz)

split_file = "/content/drive/MyDrive/project/SpeechCLIP_N_M_trial/SpeechCLIP/data/Flickr8k/Flickr_8k.trainImages.txt"
img_dir = "/content/drive/MyDrive/project/SpeechCLIP_N_M_trial/SpeechCLIP/data/Flickr8k/Images/Flicker8k_Dataset"
audio_dir = "/content/drive/MyDrive/project/SpeechCLIP_N_M_trial/SpeechCLIP/data/Flickr8k/flickr_audio/wavs"

wav_fps = []
img_list = []
# Using readlines()
file1 = open(split_file, 'r')
Lines = file1.readlines()
print("------",len(Lines))
# Strips the newline character

for line in Lines:
    # line = Lines[i]
    temp_wav = []
    flag = True
    for i in range(5):
      wav_path = os.path.join(audio_dir,line.strip()[:-4]+"_"+str(i)+".wav")
      if not(os.path.exists(wav_path)):
        flag = False
        break
      temp_wav.append(wav_path)

    img_path = os.path.join(img_dir,line.strip())
    if (os.path.exists(img_path)) and flag:
       img_list.append(img_path)
       wav_fps += temp_wav

#file_names for img
saved_file_names = "/content/drive/MyDrive/project/SpeechCLIP_N_M_trial/SpeechCLIP/output/train_images_for_GAN88_Lparallel_check.pickle"
file_names = []
for i in range(len(img_list)):
    ele = img_list[i].replace('Images/Flicker8k_Dataset/', '')
    file_names.append(ele.replace('.jpg', ''))

if not os.path.exists(saved_file_names):
  print("saving images file names ..")
  with open(saved_file_names, "wb") as f:
    pickle.dump(file_names, f)
else:
  print("images file names already exists, skip ..")


print("# images = ", len(img_list))
# print("12829: ", img_list[12829])
print("# speech = ", len(wav_fps))


#loading wave data
wav_data = []
saved_wav_data = "/content/drive/MyDrive/project/SpeechCLIP_N_M_trial/SpeechCLIP/wav_data_train88_Lparallel.pickle"
if os.path.exists(saved_wav_data):
  print("Found saved wav data")
  with open (saved_wav_data,'rb') as pick:
    wav_data = pickle.load(pick)
else:
  print("Couldn't find wav data, re-create ..")
  for _w in wav_fps:
      wav_data.append(
          torch.FloatTensor(librosa.load(_w, sr=16_000)[0]).to(device)
      )

  with open(saved_wav_data, "wb") as f:
            pickle.dump(wav_data, f)


print("done loading speech")
print("len(wav_data) = ", len(wav_data))


with torch.no_grad():
    print("start torch.no_grad()")
    # save image embeddings
    # with open("/content/drive/MyDrive/project/SpeechCLIP_N_M_trial/SpeechCLIP/output/image_output_embedding88_train_Lparallel.pickle", "wb") as f:
    #   image_tensor = model_img.prep_image(img_list)
    #   image_rep = model_img.encode_image(image_tensor)
    #   pickle.dump(image_rep, f)
    #   print("done saving images")

    #clear memory of image data
    # del image_tensor, image_rep
    gc.collect() 
    torch.cuda.empty_cache()

    # save speech embeddings
    print("starting at : ", start)
    for i in range(12904,12905,1):
      print(" *** data from ", i)
      # print("len = ", len(wav_data[batch*i : batch*(i+1)]))
      speech_rep = model.encode_speech(wav=[wav_data[i]])
      # print("model size: ",sys.getsizeof(model))
      # print(asizeof.asizeof(model)) 
      # print(speech_rep['parallel_audio_feat'].shape)
      # print("speech size", sys.getsizeof(speech_rep))
      # time.sleep(10)
      # speech_rep = np.vstack((speech_rep, temp_speech_rep['parallel_audio_feat'].cpu().detach().numpy()))
    # print(len(speech_rep), type(speech_rep), speech_rep.shape)
      with open("/content/drive/MyDrive/project/SpeechCLIP_N_M_trial/SpeechCLIP/output/speech_output_embedding_train88_Lparallel_9.pickle", "ab") as f:
        pickle.dump(speech_rep['parallel_audio_feat'], f)

      print(" **************** done ", i, " *****************")
      # clear memory of speech data
      del speech_rep
      gc.collect() 
      torch.cuda.empty_cache()
      # time.sleep(10)
    print("done saving speech")
    









