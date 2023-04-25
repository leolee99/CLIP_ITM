"""
# A simple pytorch implementation of baseline based-on CLIP for Image-text Retrieval.
#
# Writen by Hao Li, 2023
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from torch import optim
from util import set_seed_logger, get_logger
from params import parse_args
from scheduler import cosine_lr
from eval import evaluate
from dataloader.dataloaders import prepare_coco_dataloaders

global logger

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def main():
    global logger
    args = parse_args()

    seed = set_seed_logger(args)
    dir_path = os.path.join(args.checkpoint_path, args.experiments)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    logger = get_logger(os.path.join(dir_path, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model_clip, preprocess = clip.clip.load(args.vision_model, device=device, jit=False) #Must set jit=False for training
    # if device == "cpu":
    #     model.float()
    # else :
    #     clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    if args.resume:
        checkpoint = torch.load(args.resume)
        model = model_clip   # change this line to use yourself model
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Loaded model from {}".format(args.resume))

    else:
        model = model_clip   # change this line to use yourself model
        logger.info("Model Initialized!")

    model = model.cuda()

    dataloader = prepare_coco_dataloaders(args, args.dataset_root, preprocess, logger)

    if args.eval:
        train_dataloader = None
        train_length = 0
        args.epochs = 0
        test_dataloader, test_length = dataloader['test']
        Mn_R1 = evaluate(args, model, test_dataloader, logger)
    
    else:
        train_dataloader, train_length = dataloader['train']
        test_dataloader, test_length = dataloader['test']

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda()
    loss_txt = loss_txt.cuda()

    total_steps = train_length * args.epochs

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    #Mn_R1 = evaluate(args, model, test_dataloader, logger)

    # add your own code to track the training progress.

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_length)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", total_steps)

    best_score = 0
    for epoch in range(args.epochs):
        model.train()
        sloss = 0
        for idx, batch in enumerate(train_dataloader) :
            step = train_length * epoch + idx
            scheduler(step)

            optimizer.zero_grad()

            images, texts, _ = batch 
            
            images = images.cuda()
            texts = texts.cuda()
            
            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images),dtype=torch.long).cuda()

            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()

            sloss += float(total_loss)

            if device == "cpu":
                optimizer.step()
            else :
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            if (idx % args.display == 0) and (idx != 0):
                logger.info("Epoch: %d/%d, step:%d/%d, lr: %.8f, loss: %f", epoch + 1, args.epochs, idx, len(train_dataloader), optimizer.param_groups[0]['lr'], sloss / args.display)
                sloss = 0
        
        save_path = os.path.join(dir_path, f"epoch{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                #"step": steps,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            save_path,
        )
        logger.info("Saved checkpoint {} (epoch {})".format(save_path, epoch + 1))

        ## Run on val dataset for selecting best model.
        logger.info("Eval on val dataset")
        Mn_R1 = evaluate(args, model, test_dataloader, logger)

        if best_score <= Mn_R1:
            best_score = Mn_R1
            best_output_model_file = save_path
        logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

if __name__ == '__main__':
    main()

