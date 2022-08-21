import torch
from .TransformerModel import SLTModel 

def beam_search(beam_size, batch_size,model, frames, encoder_output , sentance_length,  bos_index, pad_index ):
    model.eval()
    # init the log prob 
    topk_log_probs = torch.zeros(batch_size, beam_size, device=encoder_output.device)
    topk_log_probs[:, 1:] = -torch.inf
    
    #crate_the_output
    predict = torch.full([batch_size,beam_size,sentance_length],pad_index,device=encoder_output.device)
    #first token of the first beam for in each sample is <bos>
    predict[:,:,0] = bos_index
    
    #init a fake mask (only zeros)  
    mask = torch.zeros(sentance_length,sentance_length, device=encoder_output.device)

    for i in range(1,sentance_length):
        beam_list = []
        for j in range(beam_size):
            words = predict[:,j,:]
            frames_padding_mask = (frames.sum(dim=-1) == 0).squeeze(-1)
            words_padding_mask = (words == pad_index)
            ew = model.word_embedding(words,pad_index)
            log_prob_bim = model.decoder(ew,encoder_output,mask, words_padding_mask, frames_padding_mask)[:,i,:]
            #adding to the prob of the rout to the probs that come out the decoder 
            log_prob_bim += topk_log_probs[:,j,None]
            beam_list.append(log_prob_bim)
        #flatting the beams and choose top beam_size 
        log_prob = torch.cat(beam_list,dim=1)
        top_beams = torch.topk(log_prob,beam_size, dim=1)

        #for the top k beams, recreating the beam number
        beams_vec = (top_beams.indices // (log_prob.shape[1]/beam_size)).type(torch.int64)
        #for the top k beams, recreating predcted class 
        nodes_value = (top_beams.indices % (log_prob.shape[1]/beam_size)).type(torch.int64)
        #crating the new predicated tensor
        predict = torch.cat([predict.index_select(1,beams_vec[:,j])[:,0,:] for j in range(beam_size)],dim=-1).view(batch_size,beam_size,-1)
        predict[:,:,i] = nodes_value
        topk_log_probs = top_beams.values
    
    return predict[:,0,:]

    





