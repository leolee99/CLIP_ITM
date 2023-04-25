import torch
from tqdm import tqdm

def evaluate(args, model, dataloader, logger):
    model.eval()
    with torch.no_grad():
        image_features = []
        text_features = []
        num_anns = dataloader.dataset.num_anns
        num_ids = len(dataloader.dataset)
        num_imgs = dataloader.dataset.img_length
        for idx, batch in enumerate(dataloader):

            images, texts, img_id = batch 
            
            images = images.cuda()
            texts = texts.cuda()

            batch_image_features = model.encode_image(images)
            batch_text_features = model.encode_text(texts)

            batch_image_features = batch_image_features / batch_image_features.norm(dim=1, keepdim=True)
            batch_text_features = batch_text_features / batch_text_features.norm(dim=1, keepdim=True)

            image_features.append(batch_image_features)
            text_features.append(batch_text_features)

            if idx % args.display == 0:
                logger.info("step:%d/%d", idx, len(dataloader))

        images_ids = torch.arange(0, num_ids, num_anns).cuda()
        image_features = torch.cat(image_features, dim=0)[images_ids]
        text_features = torch.cat(text_features, dim=0)

        sim_matrix = []
        
        for idx, image_feat in tqdm(enumerate(image_features)):
            logit_scale = model.logit_scale.exp()
            sim_line = logit_scale * image_feat @ text_features.t()

            sim_matrix.append(sim_line.unsqueeze(0).cpu())
        
        sim_matrix = torch.cat(sim_matrix, dim=0)
        label = torch.eye(num_imgs).unsqueeze(-1).repeat(1,1,num_anns).view(-1, num_ids)
        results = metric_compute(sim_matrix, label, logger)

    # ground_truth = torch.arange(len(images), dtype=torch.long).cuda()
    return results['mean_R1']


def metric_compute(sim_matrix, label, logger):
    results = {}
    # Image-to-Text
    i2t_rank_matrix = (-sim_matrix).argsort().argsort() + 1
    i2t_gt_rk_matrix = label * i2t_rank_matrix
    i2t_gt_rk_matrix[i2t_gt_rk_matrix==0] = 1e9
    i2t_min_rank = i2t_gt_rk_matrix.min(1).values

    results['i2t_R@1'] = 100 * torch.where(i2t_min_rank <= 1, 1, 0).type(torch.float32).mean()
    results['i2t_R@5'] = 100 * torch.where(i2t_min_rank <= 5, 1, 0).type(torch.float32).mean()
    results['i2t_R@10'] = 100 * torch.where(i2t_min_rank <= 10, 1, 0).type(torch.float32).mean()

    logger.info("Image-to-Text:")
    logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['i2t_R@1'], results['i2t_R@5'], results['i2t_R@10']))
    
    # Text-to-Image
    t2i_rank_matrix = (-sim_matrix.T).argsort().argsort() + 1
    t2i_gt_rk_matrix = label.T * t2i_rank_matrix
    t2i_gt_rk_matrix[t2i_gt_rk_matrix==0] = 1e9
    t2i_min_rank = t2i_gt_rk_matrix.min(1).values

    results['t2i_R@1'] = 100 * torch.where(t2i_min_rank <= 1, 1, 0).type(torch.float32).mean()
    results['t2i_R@5'] = 100 * torch.where(t2i_min_rank <= 5, 1, 0).type(torch.float32).mean()
    results['t2i_R@10'] = 100 * torch.where(t2i_min_rank <= 10, 1, 0).type(torch.float32).mean()

    logger.info("Text-to-Image:")
    logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['t2i_R@1'], results['t2i_R@5'], results['t2i_R@10']))
    
    results['mean_R1'] = (results['i2t_R@1'] + results['t2i_R@1']) / 2

    logger.info("Mean R1: {:.2f}".format(results['mean_R1']))
    
    return results
    