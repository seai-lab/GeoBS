import torch
import numpy as np

# Some models like the included location encoders only supports list or np.ndarray
# Coerce datatype from torch.Tensor to np.ndarray briefly, then turn it back after processing
# Would not cut the gradient 
def forward_with_np_array(batch_data, model):
    loc_b = batch_data.detach().cpu().numpy() #loc_b = np.array(batch_data)
    loc_b = np.expand_dims(loc_b, axis=1) #loc_b = np.expand_dims(batch_data, axis=1)
    loc_embedding = torch.squeeze(model(loc_b))
    return loc_embedding

# Only the loc encoder is trained; only location data is used. There is no image classifier, no image embedding, and no cnn_prediction_probas. Only the cnn_prediction_probas is relevant for GeoBS, and it only plays a part during testing.
def train(epochs,
          batch_count_print_avg_loss,
          dataloader,
          loc_encoder,
          params,
          criterion,
          optimizer,
          device):
    
    # - announcement -
    print(f'Training for {epochs} epochs.')

    # - training loop -
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        n = 0

        for i, data in enumerate(dataloader, 0):

            # - data -
            idx_b, img_b, loc_b, y_b = data
            loc_b, y_b = loc_b.to(device), y_b.to(device)

            # - optimizer -
            optimizer.zero_grad()

            # - location embedding and embedding loss / debiasing loss -
            # logits = loc_embedding = forward_with_np_array(batch_data = loc_b, model = loc_encoder)
            loss = criterion(model = loc_encoder, params = params, loc_feat = loc_b, loc_class = y_b, user_ids = None, inds = torch.arange(y_b.size(0)), neg_rand_type='spherical')
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            # - loss -
            if i % batch_count_print_avg_loss == batch_count_print_avg_loss - 1:
                print('[epoch %d, batch %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / batch_count_print_avg_loss))
                running_loss = 0.0
            epoch_loss += loss.item() * y_b.size(0)
            n += y_b.size(0)
        
        print(f"epoch {epoch+1} mean loss: {epoch_loss/n:.4f}")

    print(f'Training Completed.')

def train_ssi_debias(epochs,
          batch_count_print_avg_loss,
          dataloader,
          loc_encoder,
          params,
          criterion,
          debias_loss,
          debias_lambda,
          partitioner,
          perf_transformer,
          optimizer,
          device):

    print(f'Debiasing for {epochs} epochs.')

    for epoch in range(epochs):
        running_loss = 0.0

        epoch_loss = 0.0
        n = 0

        for i, data in enumerate(dataloader, 0):

            idx_b, img_b, loc_b, y_b = data
            loc_b, y_b = loc_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            # assume loc_b have [lat, long]
            #logits = loc_embedding = forward_with_np_array(batch_data = loc_b, model = loc_encoder)
            
            # inds = tensor, [0,1,2,...,batch_size-1]
            # user_ids = None because no support for it
            loss = criterion(model = loc_encoder, params = params, loc_feat = loc_b, loc_class = y_b, user_ids = None, inds = torch.arange(y_b.size(0), device=device), neg_rand_type='spherical')
            

            gbs_losses = []

            for idx in idx_b:
                neighborhood_idx = partitioner.get_neighborhood_idx(idx.item())
                if neighborhood_idx.shape[0] < 10:
                    continue

                neighborhood_points = partitioner.get_neighborhood_points(idx.item())

                loc_n, y_n = (torch.stack([dataloader.dataset[i][2].to(device) for i in neighborhood_idx]),
                                     torch.stack([dataloader.dataset[i][3].to(device) for i in neighborhood_idx]))
                # logits = loc_embedding = forward_with_np_array(batch_data = loc_n, model = loc_encoder)
                logits = loc_embedding = loc_encoder(loc_n)

                neighborhood_values = perf_transformer(logits, y_n)

                tmp_gbs_loss, _ = debias_loss(neighborhood_points, neighborhood_values)

                if tmp_gbs_loss is not None:
                    gbs_losses.append(tmp_gbs_loss[0])

            if len(gbs_losses) > 0:
                gbs_loss = torch.mean(torch.stack(gbs_losses))
                loss += debias_lambda * gbs_loss

            running_loss += loss.item()

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            if i % batch_count_print_avg_loss == batch_count_print_avg_loss - 1:
                print(
                    '[epoch %d, batch %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / batch_count_print_avg_loss))

                running_loss = 0.0

            epoch_loss += loss.item() * y_b.size(0)
            n += y_b.size(0)

        print(f"epoch {epoch + 1} mean loss: {epoch_loss / n:.4f}")

    print(f'Debiasing Completed.')


def train_sri_debias(epochs,
                     batch_count_print_avg_loss,
                     dataloader,
                     loc_encoder,
                     criterion,
                     params,
                     debias_loss,
                     debias_lambda,
                     partitioner,
                     partition_mode, # Choose from ScaleGrid, DistanceLag and DirectionSector
                     scale_grid,
                     distance_lag,
                     split_number,
                     perf_transformer,
                     optimizer,
                     device):
    print(f'Debiasing for {epochs} epochs.')

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
            # logits = loc_embedding = forward_with_np_array(batch_data = loc_b, model = loc_encoder)

            loss = criterion(model = loc_encoder, params = params, loc_feat = loc_b, loc_class = y_b, user_ids = None, inds = torch.arange(y_b.size(0)), neg_rand_type='spherical')

            gbs_losses = []

            for idx in idx_b:
                if partition_mode == "ScaleGrid":
                    partition_idx_list, neighborhood_idx = partitioner.get_scale_grid_idx(idx.item(), scale=scale_grid)
                elif partition_mode == "DistanceLag":
                    partition_idx_list, neighborhood_idx = partitioner.get_distance_lag_idx(idx.item(), lag=distance_lag)
                elif partition_mode == "DirectionSector":
                    partition_idx_list, neighborhood_idx = partitioner.get_direction_sector_idx(idx.item(), n_splits=split_number)
                else:
                    assert False, "Unknown Partition Mode. Please choose from ScaleGrid, DistanceLag and DirectionSector."

                if neighborhood_idx.shape[0] < 50:
                    continue

                loc_n, y_n = (torch.stack([dataloader.dataset[i][2].to(device) for i in neighborhood_idx]),
                                     torch.stack([dataloader.dataset[i][3].to(device) for i in neighborhood_idx]))

                logits = loc_embedding = loc_encoder(loc_n)

                neighborhood_values = perf_transformer(logits, y_n)
                # print("Neighborhood hist", neighborhood_values)

                for partition_idx in partition_idx_list:
                    partition_values = perf_transformer(logits[partition_idx], y_n[partition_idx])
                    # print("Partition hist", partition_values)
                    gbs_losses.append(debias_loss(partition_values, neighborhood_values))

            if len(gbs_losses) > 0:
                gbs_loss = torch.sum(torch.stack(gbs_losses))
                # print("GBS loss: {}".format(gbs_loss.item()))
                loss += debias_lambda * gbs_loss

            running_loss += loss.item()

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            if i % batch_count_print_avg_loss == batch_count_print_avg_loss - 1:
                print(
                    '[epoch %d, batch %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / batch_count_print_avg_loss))

                running_loss = 0.0

            epoch_loss += loss.item() * y_b.size(0)
            n += y_b.size(0)

        print(f"epoch {epoch + 1} mean loss: {epoch_loss / n:.4f}")

    print(f'Debiasing Completed.')