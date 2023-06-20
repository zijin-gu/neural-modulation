import argparse
import time
import torch
from torch import nn
import numpy as np
import os
from encoding_tools import load_encoding
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int, truncated_noise_sample)
import matplotlib.pyplot as plt

ROIs_subj = []
ROIs1 = ['OFA', 'FFA1', 'FFA2', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs2 = ['OFA', 'FFA1', 'FFA2', 'aTLfaces', 'EBA', 'FBA2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs3 = ['OFA', 'FFA1', 'FFA2', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs4 = ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs5 = ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs6 = ['OFA', 'FFA1', 'FFA2', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs7 = ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA2', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs8 = ['OFA', 'FFA1', 'FFA2', 'mTLfaces', 'aTLfaces', 'EBA', 'FBA1', 'FBA2', 'mTLbodies', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA1', 'VWFA2', 'mfswords', 'mTLwords', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']
ROIs_subj.append(ROIs1)
ROIs_subj.append(ROIs2)
ROIs_subj.append(ROIs3)
ROIs_subj.append(ROIs4)
ROIs_subj.append(ROIs5)
ROIs_subj.append(ROIs6)
ROIs_subj.append(ROIs7)
ROIs_subj.append(ROIs8)

def get_args():
	
	# Init a parser.
	parser = argparse.ArgumentParser (
		prog='Synthesize', 
		description='Provided a ROI ID in Range=[0, 24], synthesize an image through the NeuroGen model'
	)
	
	# Add arguments to parser.
	parser.add_argument('--roi', type=str, help='Synthesize images of this ROI.') 
	parser.add_argument('--subj', type=int, help='Subject ID.')    
	parser.add_argument('--exptype', type=str, help='maxsyn or randsyn.')        
	parser.add_argument('--steps', type=int, default=1000, help='The number of steps that should be taken to find the code.')
	parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
	parser.add_argument('--reptime', type=int, default=10, help='Number of random initializations')    
	args = parser.parse_args()

	return args


def save_image(img, cate, outdir):

	plt.figure()
	plt.imshow(img, aspect='equal')
	plt.tight_layout()
	plt.axis('off')
	plt.imsave(outdir + 'img%d'%cate + '.png', img, format='png')

	return


def center_crop(x, current_size, desired_size):
    start = int((current_size - desired_size)/2)
    return x[:,:, start:(start + desired_size), start:(start + desired_size)]


def synthesize(exptype, model, classifier, maps, weight, num_class, roi, num_steps=300, lr=0.005, wdecay=0.0001, code_len=4096, dims=(227, 227, 3), device=torch.device("cpu"), repeat_time=None):
	
  truncation = 0.4
  
	# Init a random code to start from.
	code = truncated_noise_sample(batch_size=1, truncation=truncation, seed=repeat_time)
	code = torch.from_numpy(code).to(device) 
	class_vector =  one_hot_from_int([num_class], batch_size=1)
	class_vector = torch.from_numpy(class_vector).to(device) 
	weight = torch.from_numpy(weight).float().to(device)
	optimizer = torch.optim.Adam([code.requires_grad_()], lr=lr)
	
	# Make sure we're in evaluation mode.
	model.eval()
	for i in range(len(classifier)):
		classifier[i].eval()
		maps[i].eval()

	step = 0
	meta = 0
	keep_imgs = []
	step_loss = []
	keep_act = [] 
	keep_code = []    
	while step < num_steps:
		step += 1
		meta = 0

		def closure():

			optimizer.zero_grad()

			# Produce an image from the code and the conditional class.          
			y = model(code, class_vector, truncation)
			# Normalize said image s.t. values are between 0 and 1.
			y = (y + 1.0 ) / 2.0
			y = center_crop(y, 256, 227)           
			# Try to classify this image
			pred = torch.zeros(len(classifier)).to(device) 
			for i in range(len(classifier)):            
				pred[i] = classifier[i](maps[i](y))[roi[i]]
			out = torch.matmul(weight[:-1], pred) + weight[-1] # for linear ensemble personalization
			#out = torch.mean(pred) # for group          

			# Get the loss with L2 weight decay.
			if exptype == 'maxsyn':
				loss = -out + wdecay * torch.sum( code**2 )
			elif exptype == 'randsyn':
				loss = abs(out) + wdecay * torch.sum( code**2 )
      else:
        raise ValueError
			loss.backward()

			print("Step %d"%step, 
				"\n   loss  = {}".format(loss.data), 
				"\n   act   = {}".format(out.data), 
				"\n   code  = {}".format(code[0,:5].data))

			return loss
		
		optimizer.step(closure)
		step_loss.append(closure().data)

    keep_act.append(out.cpu().detach().numpy())
    y = np.moveaxis(y.cpu().detach().numpy()[0], 0, -1)
    keep_imgs.append(y)
    keep_code.append(code.cpu().detach().numpy()[0])

	if exptype == 'maxsyn':
		opt_step = np.argmax(keep_act)
	elif exptype == 'randsyn':
		opt_step = np.argmin(list(map(abs, keep_act)))
	out_img = keep_imgs[opt_step]
	out_code = keep_code[opt_step]
	out_act = keep_act[opt_step]    
    
	print("Optimal step is ", opt_step)
	print("Optimal act  is ", out_act)
	return out_img, keep_imgs, keep_act, out_code, out_act

def main():
	# Pull some arguments from the CL.
	args = get_args()
	if args.roi not in ROIs_subj[args.subj-1]:
		raise ValueError
	now = time.ctime()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	subject = args.subj
  
  # load linear ensemble weights
	allsubj_params = np.load('./newdata/linearparams/ng%03d/%s_params.npy'%(subject, args.roi))
	allsubj_params = np.mean(allsubj_params, axis=0)    
  coef = allsubj_params[:-1].reshape(1,-1)
	intercept = allsubj_params[-1]
  
  # identify optimal classes
	all_act = []
	for trn_subject in range(1,9):
		if args.roi not in ROIs_subj[trn_subject-1]:
			continue
		else:
			roi_idx = ROIs_subj[trn_subject-1].index(args.roi)
			all_act.append(np.load('../nsd/img/S%d'%trn_subject + '/all_activations.npy')[:,:,roi_idx])
	all_act = np.mean(np.array(all_act), axis=1)
	roi_act = np.matmul(coef, all_act).reshape(-1) + intercept
	#roi_act = np.mean(all_act, axis=0)  
	if args.exptype == 'maxsyn':     
		l2h_idx = np.argsort(roi_act)
		h2l_idx = l2h_idx[::-1]
	elif args.exptype == 'randsyn':
		h2l_idx = np.argsort(abs(roi_act))
    
  # load encoding models
	fwrf, fmaps = [], []
	roi_idx = []
	for trn_subject in range(1,9):
		tmp0, tmp1 = load_encoding(subject=trn_subject, model_name='dnn_fwrf', device=device)
		fwrf.append(tmp0)
		fmaps.append(tmp1)
		if args.roi not in ROIs_subj[trn_subject-1]:
			continue
		else:
			roi_idx.append(ROIs_subj[trn_subject-1].index(args.roi))
  
  # load generator
	model = BigGAN.from_pretrained('biggan-deep-256')
	model.to(device)

	#start
	all_act = np.zeros(50)
	for cate in range(50):
		begin = time.time()
		sim_image, sim_video, keep_act, final_code, final_act = synthesize(
			exptype=args.exptype,            
			model=model, 
			classifier=fwrf,
			maps=fmaps, 
			weight=allsubj_params,
			num_class=h2l_idx[cate],
			roi=roi_idx,
			num_steps=args.steps,
			lr=args.lr, 
			wdecay=0.001,
			device=device)
		end = time.time()

		# Let me know when things have finished processing.
		print('[INFO] Completed processing SIM in {:0.4}(s)!! Requested Class {} '.format(end - begin, args.roi)) 

		output_dir = './newdata/img/ng%03d'%subject + '/' + args.roi + '/%s/'%args.exptype
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		# Save/Show the image.
		save_image(img=sim_image, cate=cate, outdir=output_dir)

		all_act[cate] = final_act
	np.save(output_dir + 'activations.npy', all_act)

if __name__ == "__main__":
	main()
