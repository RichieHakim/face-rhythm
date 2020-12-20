from face_rhythm.util import helpers

def makeDistanceMatrix(n , centerIdx , vid_height , vid_width):
    x,y = np.meshgrid(range(vid_width),range(vid_height)) # note dim 1:X and dim 2:Y
    return np.sqrt((y-int(centerIdx[1]))**2+(x-int(centerIdx[0]))**2)

def create_kernel(config_filepath):
    config = helpers.load_config(config_filepath)
    width_cosKernel = config['cdr_width_cosKernel']
    num_dots = config['cdr_num_dots'] 
    cosKernel = np.zeros((vid_height , vid_width , num_dots))
    cosKernel_mean = np.zeros(num_dots)
    for ii in range(num_dots):
        x = makeDistanceMatrix(width_cosKernel, np.squeeze(pointInds_toUse)[ii],
                               vid_height, vid_width)
        x_norm = x / width_cosKernel
        x_clipped = np.minimum(x_norm , 1)
        cosKernel[:,:,ii] = (np.cos(x_clipped * np.pi) +1) / 2
        tmp = copy.deepcopy(cosKernel[:,:,ii])
        tmp[tmp==0] = np.nan
        cosKernel_mean[ii] = np.nanmean(tmp)
    return cos_Kernel


def space_points(config_filepath, pts_all):
    config = helpers.load_config(config_filepath)
    spacing = config['cdr_spacing'] 
    
    bbox_subframe_displacement = pts_all['bbox_subframe_displacement']
    pts_displacement = pts_all['pts_displacement']
    pts_x_displacement = pts_all['pts_x_displacement']
    pts_y_displacement = pts_all['pts_y_displacement']

    pts_spaced_convDR = np.ones((np.int64(bbox_subframe_displacement[3] * bbox_subframe_displacement[2] / spacing) ,2)) * np.nan ## preallocation
    cc = 0 ## set idx counter
    # make spaced out points
    for ii in range(len(pts_x_displacement)):
        if (pts_x_displacement[ii]%spacing == 0) and (pts_y_displacement[ii]%spacing == 0):
            pts_spaced_convDR[cc,0] = pts_x_displacement[ii]
            pts_spaced_convDR[cc,1] = pts_y_displacement[ii]
            cc = cc+1

    pts_spaced_convDR = np.expand_dims(pts_spaced_convDR,1).astype('single')
    pts_spaced_convDR = np.delete(pts_spaced_convDR , np.where(np.isnan(pts_spaced_convDR[:,0,0])) , axis=0)
    return pts_spaced_convDR


def points_show(config_filepath, pts_all, pts_spaced_convDR):
    config = helpers.load_config(config_filepath)
    vidNum_toUse = config['cdr_vidNum']
    frameNum_toUse = config['cdr_frameNum']
    dot_size = config['cdr_dot_size']
    
    pts_x_displacement = pts_all['pts_x_displacement']
    pts_y_displacement = pts_all['pts_y_displacement']
    
    ## Define random colors for points in cloud
    color_tuples =  list(np.arange(len(pts_x_displacement)))
    for ii in range(len(pts_x_displacement)):
        color_tuples[ii] = (np.random.rand(1)[0]*255, np.random.rand(1)[0]*255 , np.random.rand(1)[0]*255)

    vid = imageio.get_reader(path_vid_allFiles[vidNum_toUse],  'ffmpeg')
    frameToSet = 0
    frame = vid.get_data(frameNum_toUse) # Get a single frame to use as the first 'previous frame' in calculating optic flow
    pointInds_tuple = list(np.arange(pts_spaced_convDR.shape[0])) 
    for ii in range(pts_spaced_convDR.shape[0]):
        pointInds_tuple[ii] = tuple(np.squeeze(pts_spaced_convDR[ii,0,:]))
        cv2.circle(frame,pointInds_tuple[ii], dot_size, color_tuples[ii], -1)
    cv2.imshow('dots for conv dim red',frame)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def makeConvDR(ii):
#     print(ii)
    influence_weightings = cosKernel[int(dots_new[ii][0][1]) , int(dots_new[ii][0][0]) , :]
    
    idx_nonZero = np.array(np.where(influence_weightings !=0))[0,:]

    displacements_preConvDR_x = input_traces[idx_nonZero , 0 , :] * influence_weightings[idx_nonZero][:,None]
    displacements_preConvDR_x = displacements_preConvDR_x - np.mean(displacements_preConvDR_x , axis=1)[:,None]
    displacements_preConvDR_y = input_traces[idx_nonZero , 1 , :] * influence_weightings[idx_nonZero][:,None]
    displacements_preConvDR_y = displacements_preConvDR_y - np.mean(displacements_preConvDR_y , axis=1)[:,None]
    pca.fit(displacements_preConvDR_x)
    output_PCA_loadings0 = pca.components_.T
    pca.fit(displacements_preConvDR_y)
    output_PCA_loadings1 = pca.components_.T
    
    output_PCA_scores0 = np.dot( displacements_preConvDR_x  ,  output_PCA_loadings0 )
    output_PCA_scores1 = np.dot( displacements_preConvDR_y  ,  output_PCA_loadings1 )
    positions_convDR_meanSub = np.zeros((2,input_traces.shape[2]))
    positions_convDR_meanSub[0,:] = np.mean(np.dot( output_PCA_loadings0[:,:rank_reduced] , output_PCA_scores0[:,:rank_reduced].T ) , axis=1) / cosKernel_mean[ii]
    positions_convDR_meanSub[1,:] = np.mean(np.dot( output_PCA_loadings1[:,:rank_reduced] , output_PCA_scores1[:,:rank_reduced].T ) , axis=1) / cosKernel_mean[ii]
    return positions_convDR_meanSub

def compute_influence(config_filepath, pointInds_toUse, pts_spaced_convDR, cosKernel, cosKernel_mean):
    config = helpers.load_config(config_filepath)
    num_components = config['cdr_num_components'] 
    
    input_traces = np.float32(positions_new_sansOutliers)
    num_components = 2
    rank_reduced = num_components

    dots_old = pointInds_toUse
    dots_new = pts_spaced_convDR

    pca = sk.decomposition.PCA(n_components=num_components)
    # p = Pool(multiprocessing.cpu_count())
    p = Pool(int(multiprocessing.cpu_count()/3))
    positions_convDR_meanSub_list = p.map(makeConvDR , range(dots_new.shape[0]))
    p.close()
    p.terminate()
    p.join()

    positions_convDR_meanSub = np.zeros((dots_new.shape[0] , 2 , input_traces.shape[2]))
    for ii in range(dots_new.shape[0]):
        positions_convDR_meanSub[ii,:,:] = positions_convDR_meanSub_list[ii]
        
    return positions_convDR_meanSub


def display_displacements(config_filepath, positions_convDR_meanSub, pts_spaced_convDR):
    config = helpers.load_config(config_filepath)

    # positions_toUse = positions_new_absolute_sansOutliers
    positions_toUse = (positions_convDR_meanSub + np.squeeze(pts_spaced_convDR)[:,:,None])

    # vidNums_toUse = range(numVids) ## note zero indexing!
    vidNums_toUse = range(3) ## note zero indexing!

    if type(vidNums_toUse) == int:
        vidNums_toUse = np.array([vidNums_toUse])

    dot_size = config['cdr_dot_size']
    printFPS_pref = config['printFPS_pref']
    fps_counterPeriod = config['fps_counterPeriod'] ## number of frames to do a tic toc over


    ## Define random colors for points in cloud
    color_tuples =  list(np.arange(positions_toUse.shape[0]))
    for ii in range(positions_toUse.shape[0]):
        color_tuples[ii] = (np.random.rand(1)[0]*255, np.random.rand(1)[0]*255 , np.random.rand(1)[0]*255)
    #     color_tuples[ii] = (0,255,255)
        

    ## Main loop to pull out displacements in each video   
    ind_concat = int(np.hstack([0 , np.cumsum(numFrames_allFiles)])[vidNums_toUse[0]])

    fps = 0
    tic_fps = time.time()
    for iter_vid , vidNum_iter in enumerate(vidNums_toUse):
        path_vid = path_vid_allFiles[vidNum_iter]
        vid = imageio.get_reader(path_vid,  'ffmpeg')

        video = cv2.VideoCapture(path_vid)
        numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for iter_frame , new_frame in enumerate(vid):
            for ii in range(positions_toUse.shape[0]):
                pointInds_tracked_tuple = tuple(np.int64(np.squeeze(positions_toUse[ii,:,ind_concat])))
                cv2.circle(new_frame,pointInds_tracked_tuple, dot_size, color_tuples[ii], -1)

            cv2.putText(new_frame, f'frame #: {iter_frame}/{numFrames}-ish', org=(10,20), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
            cv2.putText(new_frame, f'vid #: {iter_vid+1}/{len(vidNums_toUse)}', org=(10,40), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
            cv2.putText(new_frame, f'total frame #: {ind_concat+1}/{numFrames_total_rough}-ish', org=(10,60), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
            cv2.putText(new_frame, f'fps: {np.uint32(fps)}', org=(10,80), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
            cv2.imshow('post outlier removal',new_frame)


            k = cv2.waitKey(1) & 0xff
            if k == 27 : break

            ind_concat = ind_concat+1


            if ind_concat%fps_counterPeriod==0:
                elapsed = time.time() - tic_fps
                fps = fps_counterPeriod/elapsed
                if printFPS_pref:
                    print(fps)
                tic_fps = time.time()
            

    cv2.destroyAllWindows()