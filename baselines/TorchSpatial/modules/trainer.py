import torch
import numpy as np

# Some models like the included location encoders only supports list or np.ndarray
# Coerce datatype from torch.Tensor to np.ndarray briefly, then turn it back after processing
def forward_with_np_array(batch_data, model):
    loc_b = batch_data.detach().cpu().numpy() #loc_b = np.array(batch_data)
    loc_b = np.expand_dims(loc_b, axis=1) #loc_b = np.expand_dims(batch_data, axis=1)
    loc_embedding = torch.squeeze(model(coords = loc_b))
    return loc_embedding

def train(task,
            epochs, 
            batch_count_print_avg_loss,
            dataloader,
            loc_encoder,
            decoder,
            criterion,
            optimizer, 
            scheduler,
            device):
    
    decoder = decoder.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        
        epoch_loss = 0.0
        n = 0

        for i, data in enumerate(dataloader, 0):
            
            idx_b, img_b, loc_b, y_b = data
            img_b, loc_b, y_b = img_b.to(device), loc_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            # assume loc_b have [lat, long]
            img_embedding = img_b
            loc_embedding = forward_with_np_array(batch_data = loc_b, model = loc_encoder)

            if task == "Classification":
                loc_img_interaction_embedding = torch.mul(loc_embedding, img_embedding)
                logits = decoder(loc_img_interaction_embedding)
        
                loss = criterion(logits, y_b)

            elif task == "Regression":
                loc_img_concat_embedding = torch.cat((loc_embedding, img_embedding), dim=1)

                yhat = decoder(loc_img_concat_embedding).squeeze(-1)

                loss = criterion(yhat, y_b) # training on standardized yhat
            
            running_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            if i % batch_count_print_avg_loss == batch_count_print_avg_loss - 1:
                print('[epoch %d, batch %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / batch_count_print_avg_loss))

                running_loss = 0.0

            epoch_loss += loss.item() * y_b.size(0)
            n += y_b.size(0)
        
        print(f"epoch {epoch+1} mean loss: {epoch_loss/n:.4f}")
        scheduler.step(epoch_loss/n)

        

    print(f'Training Completed.')

def train_debias(task,
          epochs,
          batch_count_print_avg_loss,
          dataloader,
          loc_encoder,
          decoder,
          criterion,
          debias_loss,
          debias_lambda,
          partitioner,
          perf_transformer,
          optimizer,
          scheduler,
          device):
    decoder = decoder.to(device)

    for epoch in range(epochs):
        running_loss = 0.0

        epoch_loss = 0.0
        n = 0

        for i, data in enumerate(dataloader, 0):

            idx_b, img_b, loc_b, y_b = data
            img_b, loc_b, y_b = img_b.to(device), loc_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            # assume loc_b have [lat, long]
            img_embedding = img_b
            loc_embedding = forward_with_np_array(batch_data=loc_b, model=loc_encoder)

            if task == "Classification":
                loc_img_interaction_embedding = torch.mul(loc_embedding, img_embedding)
                logits = decoder(loc_img_interaction_embedding)

                loss = criterion(logits, y_b)

                gbs_losses = []

                for idx in idx_b:
                    neighborhood_idx = partitioner.get_neighborhood_idx(idx.item())
                    if neighborhood_idx.shape[0] < 10:
                        continue

                    neighborhood_points = partitioner.get_neighborhood_points(idx.item())
                    # background_points = partitioner.get_background_points(idx.item())

                    img_n, loc_n, y_n = (torch.stack([dataloader.dataset[i][1].to(device) for i in neighborhood_idx]),
                                         torch.stack([dataloader.dataset[i][2].to(device) for i in neighborhood_idx]),
                                         torch.stack([dataloader.dataset[i][3].to(device) for i in neighborhood_idx]))

                    img_embedding = img_n
                    loc_embedding = forward_with_np_array(batch_data=loc_n, model=loc_encoder)

                    loc_img_interaction_embedding = torch.mul(loc_embedding, img_embedding)
                    logits = decoder(loc_img_interaction_embedding)

                    neighborhood_values = perf_transformer(logits, y_n)
                    # background_values = torch.zeros(background_points.shape[0])

                    # points = np.concatenate([neighborhood_points, background_points], axis=0)
                    # values = torch.cat([neighborhood_values, background_values])

                    gbs_losses.append(debias_loss(neighborhood_points, neighborhood_values)[0])

                    # gbs_losses.append(debias_loss(points, values)[0])

                gbs_loss = torch.mean(torch.stack(gbs_losses))
                # print("Original Loss: {}, GBS Loss: {}".format(loss.item(), gbs_loss.item()))

                loss += debias_lambda * gbs_loss

            elif task == "Regression":
                loc_img_concat_embedding = torch.cat((loc_embedding, img_embedding), dim=1)

                yhat = decoder(loc_img_concat_embedding).squeeze(-1)

                loss = criterion(yhat, y_b)  # training on standardized yhat

            running_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            if i % batch_count_print_avg_loss == batch_count_print_avg_loss - 1:
                print(
                    '[epoch %d, batch %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / batch_count_print_avg_loss))

                running_loss = 0.0

            epoch_loss += loss.item() * y_b.size(0)
            n += y_b.size(0)

        print(f"epoch {epoch + 1} mean loss: {epoch_loss / n:.4f}")
        scheduler.step(epoch_loss / n)

    print(f'Debiasing Completed.')