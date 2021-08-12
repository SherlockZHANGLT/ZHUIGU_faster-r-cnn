import numpy as np
import scipy.misc
import scipy.ndimage
import random
from skimage.transform import radon
#from keras.utils import to_categorical
from skimage import measure
import cv2
import torch

def feed_placeholder(dataset, batch_size, given_ids, anchors, config, num_rois, input_image, input_image_meta, input_rpn_match,
                     input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks, input_gt_1_hot, input_sparse_gt, disp,
                     is_training):
    """
    占位符赋值。
    注意应该是在这里执行MaskRCNN_0_get_inputs.get_image_ids_for_next_batch这些函数，
        从dataset里得到这一批次的input_image_real、input_image_meta_real等。
    """
    if given_ids:  # 这种情况就是指定某两张图用来训练的，一般用于训练一两步检查网络的时候。
        assert len(given_ids) == batch_size
        image_id_selected = given_ids
        real_patient_id = []
        for i in image_id_selected:
            patiend_id_this = dataset.image_info[i]['patient']
            real_patient_id.append(patiend_id_this)
        real_patient_id = np.array(real_patient_id)
    else:
        image_id_selected, real_patient_id = get_image_ids_for_next_batch(dataset, batch_size,
                                                                                                shuffle=is_training)
        # 上句，复用网络的时候加了个shuffle=flip。
    if disp:
        print('当前已经完成的时代数：%d' % dataset._epochs_completed)
        print('当前批次开始的索引号：%d' % (dataset._index_in_epoch - batch_size))  # 现在dataset._index_in_epoch是当前批次结束的索引号，所以减一下。
        print('选择了以下这几个序号的图作为本批次的训练集：', image_id_selected, '真实病人序号是：', real_patient_id)
    # 上句，image_id_selected是0~119之间的，real_patient_id是实际病人序号。
    inputs_for_MRCNN_dict_feeding = get_batch_inputs_for_MaskRCNN(dataset, image_id_selected,
                                                                                        anchors, config, num_rois,
                                                                                        augment=is_training)
    input_image_real = inputs_for_MRCNN_dict_feeding[0]  # 归一化到0~255的uint8类型，考虑到这个类型确实是只能0~255，而且原来用MRCNN弄好了的那次也就是这样，就不弄到0~1了吧？。。
    input_image_meta_real = inputs_for_MRCNN_dict_feeding[1]
    input_rpn_match_real = inputs_for_MRCNN_dict_feeding[2]
    input_rpn_bbox_real = inputs_for_MRCNN_dict_feeding[3]
    input_gt_class_ids_real = inputs_for_MRCNN_dict_feeding[4]
    input_gt_boxes_real = inputs_for_MRCNN_dict_feeding[5]
    input_gt_masks_real = inputs_for_MRCNN_dict_feeding[6]  # MRCNN用的每个器官的掩膜
    input_gt_sparse_radon = inputs_for_MRCNN_dict_feeding[8]  # radon变换稀疏表示金标准
    gt_boxes_and_labels = np.concatenate([input_gt_boxes_real / 512, np.expand_dims(input_gt_class_ids_real, axis=2)],
                                         axis=2)
    gt_trim = batch_processing \
        (process_func=trim_gt, input_batch=gt_boxes_and_labels, num_classes=config.NUM_CLASSES)
    y_gt = gt_trim[:, :, 4]
    y_gt_one_hot_real = to_categorical(y_gt, config.NUM_CLASSES)  # 把标签变成1热形式。
    feed_dict = {input_image: input_image_real, input_image_meta: input_image_meta_real,
                 input_rpn_match: input_rpn_match_real, input_rpn_bbox: input_rpn_bbox_real,
                 input_gt_class_ids: input_gt_class_ids_real, input_gt_boxes: input_gt_boxes_real,
                 input_gt_masks: input_gt_masks_real, input_gt_1_hot: y_gt_one_hot_real, input_sparse_gt: input_gt_sparse_radon}
    # 上句中的input_gt_boxes: input_gt_boxes_real是说，赋给input_gt_boxes的是input_gt_boxes_real（还没有归一化的金标准外接矩形）
    """【基础】似乎sess.run和feed_dict的时候，不是所有的赋值都要用得上的。
    就像第一步run那个train_op_MRCNN，这一步不需要input_gt_mask，但是给feed进去也没啥的。。"""
    return feed_dict, image_id_selected, real_patient_id, input_image_real, \
           input_gt_class_ids_real, input_gt_boxes_real, input_gt_masks_real

def get_image_ids_for_next_batch(dataset, batch_size, shuffle=True):
    """【基础】按照索引号，取出一个批次的图像。
    注意这个函数的写法，很有用！！主要是要设定dataset的三个成员变量：_index_in_epoch、_epochs_completed、_image_ids_shuffle，
        分别代表当前时代下，弄到了哪张图；当前弄到了第几个时代；当前时代下的乱序图片序号序列！
    """
    num_examples = len(dataset.image_ids)  # 数据集里的总元素个数。
    start_id = dataset._index_in_epoch  # 这个批次，从数据集里的第几个元素开始。
    image_id = np.copy(dataset.image_ids)  # 0~19的array，相当于range(num_examples)。
    # 处理第一个时代，获得打乱了顺序的“图像信息序列”。
    if dataset._epochs_completed == 0 and start_id == 0 and shuffle:  # _epochs_completed似乎应该在外面函数里设计。。
        perm0 = np.arange(num_examples)
        np.random.shuffle(perm0)
        dataset._image_ids_shuffle = image_id[perm0]  # 数据集里所有图像，按照打乱了的顺序，排成一个list。
    if start_id + batch_size <= num_examples:  # 在一个时代内取数据
        dataset._index_in_epoch += batch_size
        end_id = dataset._index_in_epoch
        image_id_selected = dataset._image_ids_shuffle[start_id:end_id]
    else:
        dataset._epochs_completed += 1  # 这个时代结束
        # 把数据集中剩下的数据弄出来
        rest_num_examples = num_examples - start_id
        image_id_rest_part = dataset._image_ids_shuffle[start_id:num_examples]
        # Shuffle the data
        if shuffle:
            perm = np.arange(num_examples)
            np.random.shuffle(perm)  # 如果shuffle，重新弄一组打乱了的数据。
            dataset._image_ids_shuffle = dataset._image_ids_shuffle[perm]
        # 开始新时代
        start_id = 0
        dataset._index_in_epoch = batch_size - rest_num_examples
        end_id = dataset._index_in_epoch
        image_id_new_part = dataset._image_ids_shuffle[start_id:end_id]
        image_id_selected = np.concatenate((image_id_rest_part, image_id_new_part), axis=0)
    """以上if-else是得到要取的那几张图在输入数据集的序号，下面取出相应的图像索引号，即在“新旧自编号对照表”中是哪个病人。"""
    dataset_patiend_ids = []
    for i in image_id_selected:
        patiend_id_this = dataset.image_info[i]['patient']
        dataset_patiend_ids.append(patiend_id_this)
    dataset_patiend_ids = np.array(dataset_patiend_ids)
    return image_id_selected, dataset_patiend_ids

def get_batch_inputs_for_MaskRCNN(dataset, image_ids, anchors, config, num_rois, augment=False):
    class_label_pos = [28, 85, 142, 199, 256, 313, 370, 427, 484]
    batch_images = []
    batch_gt_mask = []
    batch_image_meta = []
    batch_gt_class_ids = []
    batch_gt_boxes = []
    batch_gt_masks = []
    batch_rpn_match = []
    batch_rpn_bbox = []
    batch_sparse_radon = np.zeros([config.BATCH_SIZE, num_rois,config.IMAGE_SHAPE[0]*config.projection_num])
    for co, image_id in enumerate(image_ids):
        image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_mask = \
            load_image_gt(dataset, config, image_id, augment=augment, use_mini_mask=config.USE_MINI_MASK)
        """注意，上句load_image_gt中，调用了load_image函数，就把灰度图变成了RGB图了。"""
        # gt_mask是这一张图的、语义分割掩膜（用来弄那个GAN的）。注意和那个gt_masks不一样。
        if not np.any(gt_class_ids > 0):
            print("警告：在第%d张图的金标准中，没有看到任何正例！" % image_id)
            continue
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors, gt_class_ids, gt_boxes, config)
        sparse_radon = np.zeros([num_rois, config.IMAGE_SHAPE[0]*config.projection_num])  # 单张图的稀疏表示，应该是有
        for i in range(gt_boxes.shape[0]):
            gt_box = np.expand_dims(gt_boxes[i, :], axis=0)
            sparse_radon0, _ = get_sparse_gt(dataset.image_info[0]['height'], dataset.image_info[0]['width'],
                                                           gt_box, projection_num=config.projection_num)
            # 上句，radon变换，构造每一个物体的稀疏表示金标准（每个物体往4个投影轴上的投影），是(config.IMAGE_SHAPE[0](512), config.projection_num(4))的

            for j in range(config.projection_num):
                sparse_radon0[class_label_pos[9 - gt_class_ids[i]], j] = 1
            #以上，给类别标签到稀疏编码

            sparse_radon1 = np.reshape(sparse_radon0, [1, config.IMAGE_SHAPE[0] * config.projection_num])
            # 上句，因为Python的radon变换输出的是“原图大小*投影直线数”的，所以上句要展平成这样的shape。这个是没关系的，仍然是所有的投影结果共同监督的。
            sparse_radon[i, :] = sparse_radon1
        if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:  # 如果金标准物体数太多，做个下采样
            ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            print(gt_class_ids)
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]
        batch_images.append(image)
        batch_gt_mask.append(gt_mask)
        batch_image_meta.append(image_meta)
        gt_class_ids_padded_zeros = np.zeros((config.MAX_GT_INSTANCES), dtype=np.int32)
        # 上句，因为每张图gt_class_ids的个数未必相同，所以直接append可能报错，所以先补齐了0。
        gt_class_ids_padded_zeros[:gt_class_ids.shape[0]] = gt_class_ids  # 前面若干个0替换成gt_class_ids中的类别标签。
        batch_gt_class_ids.append(gt_class_ids_padded_zeros)
        gt_boxes_padded_zeros = np.zeros((config.MAX_GT_INSTANCES, 4), dtype=np.int32)  # 类似于上面
        gt_boxes_padded_zeros[:gt_boxes.shape[0]] = gt_boxes
        batch_gt_boxes.append(gt_boxes_padded_zeros)
        if config.USE_MINI_MASK:
            batch_gt_masks_padded_zeros = np.zeros((config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], config.MAX_GT_INSTANCES))  #
        else:
            batch_gt_masks_padded_zeros = np.zeros((image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))
        batch_gt_masks_padded_zeros[:,:,:gt_masks.shape[-1]] = gt_masks
        batch_gt_masks.append(batch_gt_masks_padded_zeros)
        batch_rpn_match.append(rpn_match)
        batch_rpn_bbox.append(rpn_bbox)
        batch_sparse_radon[co, :, :] = sparse_radon
    batch_images = np.stack(batch_images, axis=0)  # (batch_size, 512, 512, 3)
    batch_image_meta = np.stack(batch_image_meta, axis=0)  # (batch_size, 15)
    batch_gt_mask = np.stack(batch_gt_mask, axis=0)  #  应该是(batch_size, 512, 512)
    batch_gt_class_ids = np.stack(batch_gt_class_ids, axis=0)  # (batch_size, 100)
    batch_gt_boxes = np.stack(batch_gt_boxes, axis=0)  # (batch_size, 100, 4)
    batch_gt_masks = np.stack(batch_gt_masks, axis=0)  # (batch_size, 56, 56, 100)
    batch_rpn_match = np.stack(batch_rpn_match, axis=0)  # (batch_size, 65472)
    batch_rpn_match = np.reshape(batch_rpn_match, [batch_rpn_match.shape[0], batch_rpn_match.shape[1], 1])  # 升维变成(batch_size, 65472, 1)
    batch_rpn_bbox = np.stack(batch_rpn_bbox, axis=0)  # (batch_size, 256, 4)
    batch_sparse_radon = np.stack(batch_sparse_radon, axis=0)  # (batch_size, num_rois, 图像大小*投影数)
    inputs_for_MRCNN_dict_feeding = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
              batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, batch_gt_mask, batch_sparse_radon]
    return inputs_for_MRCNN_dict_feeding

def load_image_gt(dataset, config, image_id, augment=False,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).
    加载一张图片的金标准数据：图像、掩膜、外接矩形。
    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
        输入augment：是否进行数据增强
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.
        输入use_mini_mask：是否局部放大掩膜。

    Returns:输出：和data_generator里的一样。
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    # 上句，应该是utils里的那个load_image函数吧，但是发现image里的东西全都是0啊（其实不是的，
    #     全是0是因为我看的是View as Array里看的是image[0]，是个512*3的矩阵，这相当于是图像第一列的三个通道，
    #     由于在图像边上，当然都是0了，但是试了试弄个image[111]看看，就有好多不是0的了啊。）。。
    masks, class_ids = dataset.load_mask(image_id)
    # 上句，应该是organs_training或testing里的那个load_image函数（utils里的那个只能生成一大堆的0好像），
    #     发现masks里的东西也都是0（然而实际上不是，m = masks[:, :, 0]再看m就可以发现有一些1的），
    #     然后class_ids是1 3 4三个数（或者别的数吧）
    """以上两句，其实可以理解为，调用函数，从dataset数据结构中解包出来原图、掩膜、金标准分类。
    详细可以在mo-net.py中搜索“解包”。"""
    mask = dataset.image_info[image_id]['mask']
    shape = image.shape
    image, window, scale, padding = resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    # 上句，缩放图像。不过估计没啥影响，因为image原来就是512,512,3的，完了之后还是这个size。
    masks = resize_mask(masks, scale, padding)
    # 上句，缩放图像，输入的scale和padding是从上上句得到的。不过，同样似乎没啥影响。

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            masks = np.fliplr(masks)
            mask = np.fliplr(mask)

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding masks got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = extract_bboxes(masks)
    # 上句，从掩膜计算外接矩形。shape=(3,4)，3表示有3个通道，4表示每个通道对应4个角点坐标。

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    # 上句，active_class_ids此时是7（dataset.num_classes=7）个0。
    # source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]  # 原来的
    # 原来的程序，上句source_class_ids此时是0~6这7个数（dataset是个HaNDataset类型变量，source_class_ids是其数据成员）。
    #     然后[]里面的dataset.image_info[image_id]["source"]是说，
    #         第image_id张图中的image_info数据成员中的下标为"source"的东西，应该要么是''要么是'organs'，（后来CT数据集，应该是要么是''、要么是'CT'、要么是'MRI'）
    #     所以上句等号右边就成了dataset.source_class_ids['']或者dataset.source_class_ids['organs']，
    #         所以就要么是[0]要么是[0,1,2,3,4,5,6]。
    # utils.py里的prepare函数里说了这个。
    source_class_ids = dataset.source_class_ids['organs']  # 改的
    # 检测数据集（dataset.image_info[image_id]["source"]不存在），索性直接弄成dataset.source_class_ids['organs']了，
    #     就是说这个数据集里可能有这10中东西，即0是背景类、加上1~9这9个椎骨类别。这应该是整个数据集的，而不是某一张图有没有这个类别的。
    active_class_ids[source_class_ids] = 1
    # 上句，active_class_ids是7个1。这玩意似乎表示，当前的数据集（应该不是当前这张图）中，这7种分类都有可能存在。
    #     后面输出的那个class_ids，才是这张图里有哪几个非背景类类别的。。

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        masks = minimize_mask(bbox, masks, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, shape, window, active_class_ids)
    # 上句，拼成所谓的meta，作为class MaskRCNN的输入

    return image, image_meta, class_ids, bbox, masks, mask

def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.
    保留长宽比，缩放图像
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim
        输入的padding：如果这个是True，就补零，让图像的大小是max_dim*max_dim

    Returns:
    image: the resized image
        返回的image：放缩了的图像。
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
        返回的图像窗口：如果提供了max_dim，可能在返回的图像中做了补零。如果这样的话，这个窗口就是补零后的图像（full image）中，
            去除了补零的地方。x2和y2不算在去除补零了的地方。
    scale: The scale factor used to resize the image
        返回的scales：图像缩放因子。
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
        返回的padding：给图像加上的补零。
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding



def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.
    用给定的尺度和补零来缩放掩膜。
        通常，这个尺度和补零是从resize_image()得到的，来保证图像和掩膜被同步地缩放。
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    从掩膜计算外接矩形。
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]  # 掩膜值为1的地方。随机用一张图验证过，发现np.any(m, axis=0)是一大堆TRUE和FALSE，然后np.where(...axis=0)就是从左到右一列列地去找，找到m中有1的那些列（此例的输出为232,233,234...271），后面的[0]是个小细节，把(?,)变成()。
        vertical_indicies = np.where(np.any(m, axis=1))[0]  # 类似上上句，从上到下一行行地找，找到m中有1的那些行。
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to resizing or cropping. Set bbox to zeros
            # 如果没有掩膜，就把外接矩形全都置零。
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_mask()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use parse_image_meta() to parse the values back.
    取出图像的属性，放到1个1D向量中去。（下面那个parse_image_meta是读取这些图像属性的。）

    image_id: An int ID of the image. Useful for debugging.
        输入image_id：图像序号
    image_shape: [height, width, channels]
        输入image_shape：图像的形状
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
        输入window：补零后，原有图像的位置
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
        输入active_class_ids：图像所在的数据集中的、可用的class_ids的列表。
            如果要用多个数据集中的图像训练，而这些数据集中并没有所有的类别的时候，这就是有用的。
            好像是说，如果第一个数据集有7种类别，第二个数据集只有其中的5种，那么第二个数据集中取出来的图像，
            active_class_ids中就有两个数是0吧。
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta

def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    给定锚点和金标准外接矩形，计算重合度，并且识别正例锚和外接矩形修正，来细化锚以匹配金标准外接矩形。

    anchors: [num_anchors, (y1, x1, y2, x2)]  输入anchor：锚，shape应该是(锚数, 4)，我们这儿是(65472, 4)。
    gt_class_ids: [num_gt_boxes] Integer class IDs.  输入gt_class_ids：shape应该是金标准外接矩形数目，某个样例中是(28,)。
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]  输入gt_boxes：shape应该是（金标准外接矩形数目，4），某个样例中是(28,4)。
    以上是输入参数，都是前面utils.generate_pyramid_anchors和load_image_gt函数的输出。

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
        输出rpn_match：锚和金标准外接矩形之间的匹配：1是正例，-1是负例，0是中性。
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        输出rpn_bbox：锚外接矩形的修正值。

    第二遍又看了一遍：
    知道它是这样用的：从“金标准类别gt_class_ids”和“外接矩形gt_boxes”，到“RPN类别金标准rpn_match”和“rpn外接矩形修正rpn_bbox”。
    第二遍想弄明白什么问题？
    1、输出rpn_bbox为什么是修正，修正干什么用的？
        【就是把某个正例锚，对应到离它最近的金标准外接矩形上，然后就看锚的那个矩形和外接矩形之间的位置（和长宽）差别，予以修正。
        这一步相当于是把图中的每个金标准，对应到了离他最近的正例锚上。】
    2、为什么要把金标准类别和外接矩形弄成rpn_match和rpn_bbox？
        【就是要知道，哪些锚是正例，然后他们和金标准有什么对应关系。因为正式训练的时候，也不是直接用金标准类别和掩膜算的损失，
        比如说预测出来一个提出（弄出来它的类别和外接矩形了），但是，原图中有好几个金标准呢，电脑怎么知道这个提出应该对应哪个金标准啊！
        所以要用那个锚点，把预测出来的那个提出对应到最近的锚点（应该是正例锚点吧）上，
        然后再找离那个锚点最近的金标准（就是那些target_class_ids、target_bbox什么的），这样就可以用那个金标准去修正那个类别和外接矩形了。
        所以说那些锚就是起一个定位作用的，找到和某个预测位置上最接近的金标准。】
    3、输出的rpn_match为何是±1和0，而不是类别索引号？
        【可以解释为这儿只需要知道它是正例还是负例，然后下一步选出正例去干后面的工作。但为什么要这样做呢？】
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)  # 现在是65472个0组成的向量
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))  # 现在是shape=(256, 4)的0向量。

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    # 现在这个拥挤矩形应该更明白了。这个gt_class_ids是输入进来的金标准分类号，如果按照那个样例说的，是[1,3,4]的话，那么crowd_ix就没有。
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
        # 最终得到这个no_crowd_bool，是个shape=(65472,)的矩阵，每个值都是0或者1，如果是1的话就说明相应的锚不是拥挤矩形。
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)  # 此种情况，65472个都是1，所有的锚都不是拥挤矩形。

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(anchors, gt_boxes)
    # shape=(num_anchors,num_gt_boxes)的矩阵，每个锚和金标准外接矩形重合度。用b=np.max(overlaps)看了看，b得0.579，即最大重合度差不多是这个。

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    # 上句，结果是个(65472,)的向量，其中大部分元素是0，但是也有几个不是的。
    #     就是说，这65472个数就是这么多个锚，然后每个数的值，就表示相应的锚和第几个金标准重合度最大啊。
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    # 上句，anchor_iou_max的shape=(65472,)，数据都是0~1之间的，因为是重合度啊。
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 上句，rpn_match仍然是65472个数的向量，只不过“与金标准外接矩形重合度小于<0.3且非拥挤矩形”这样的位置中的数，都被写成了-1，
    #     这些位置对应的锚就叫做负样本。

    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)  # “和金标准外接矩形重合度最大的锚”的位置，如62929。
    rpn_match[gt_iou_argmax] = 1  # 该位置的样本设为正样本
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1
    # 上句，如果有的锚“和金标准外接矩形重合度”>0.7，那么这些锚就设为正样本。
    #     2和3两个都执行，是为了保证所有锚和金标准重合度都小于0.7的时候，也能至少有一个正样本。
    # rpn_match_is_1 = np.where(rpn_match==1)
    # if rpn_match_is_1[0].shape[0] > 3 :
    #     print('有3个以上的锚点被认为是RPN正例。')  # 看看这种情况，RPN的金标准和预测值是什么样子？写在下面。。。
    """以上，总体的目的是为了给每个金标准外接矩形，找到一个锚（Set an anchor for each GT box）。
    也就是说，要通过overlaps（包含金标准gt_boxes和锚anchor的位置信息）找到所有的正样本，
        并且，把rpn_match中、正样本索引号对应的位置标记为1。
    上面，“有3个以上的锚点被认为是RPN正例”的情况，是这样的————
        金标准：那个rpn_match就是3个以上等于1的（3个以上的RPN正例啊）了，没啥好说的。
            那个金标准RPN外接矩形修正值就是那几个RPN正例的外接矩形修正值，其他的都是0了。
            （如果有3个以上的RPN正例，那么这个金标准RPN外接矩形也会有3个以上不为0的情况了）
            另外这个RPN里的金标准外接矩形修正值，和mrcnn里的那个外接矩形修正值是不一样的。即使是训练了很长时间也是不一样的。
        预测值：RPN_class就是那65472个锚中的每一个，为负例和正例的概率。
            外接矩形就是(2, 65472, 4)，是这65472个都有的，然后我发现预测为正例的那几个，确实是会逐渐接近于金标准的外接矩形修正值。
    """

    # Subsample to balance positive and negative anchors  下采样，平衡正例和负例
    # Don't let positives be more than half the anchors  不要让正例多于锚数目的一半
    ids = np.where(rpn_match == 1)[0]  # 正例的索引，如刚才的那个62929。
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    # 多了的正样本的个数。就是正样本数len(ids)-总样本数的一半。现在由于正样本就1个，所以这个extra是个负数。
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0# 这是随机地把多余的正样本弄成中性样本。当然上面的情况，这个if里的东西都没执行。
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]  # 负例的索引，一般都有一大堆的负例。比如现在就有65460个（说明这个样本负例严重超了啊）
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    # 上句，负例本来应该有config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1)（即总样本数-正例数）个，实际上有len(ids)个，就多了一些。
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0  # 把多余的负例弄成中性的了

    # For positive anchors, compute shift and scale needed to transform them to match the corresponding GT boxes.
    # 对于每个正例样本，计算位移和放缩量，变换他们，让他们和对应的金标准外接矩形匹配。
    ids = np.where(rpn_match == 1)[0]  # 正例样本索引号。
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):  # 这个for循环，i是ids中的一个数（即某个正例的索引号），anchors[ids]就是该正例（即锚）的角点坐标。
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]  #
        """上句，找到离当前正例（第i个正例）最近的金标准外接矩形。
        注意i只是在ids里循环，也就是说只是正例的序号，如62929、62930等等，而并不是从0~65471的那些。
        然后，anchor_iou_argmax[i]就表示，离第i个样本（因为i属于ids，所以这个样本肯定是正例）最近的金标准外接矩形。
        """

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w  # 以上，计算出来金标准和锚（即那个正例）的长宽、中心点坐标。

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [(gt_center_y - a_center_y) / a_h, (gt_center_x - a_center_x) / a_w, np.log(gt_h / a_h), np.log(gt_w / a_w),]
        # 上句，计算出来外接矩形修正的那些数。
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV  # 正则化
        ix += 1
        """
        以上，把锚匹配到金标准外接矩形中，即先从65472个锚中把正例选出来，然后找到其中的正例对应的金标准外接矩形，最终计算出需要修正的值（rpn_bbox）
        忽然想起来一个事儿，class MaskRCNN里的【从图片到提出的第五步】，
            在detection_targets_graph函数里也去除了拥挤矩形，然后弄了正例负例什么的。这里有什么不一样的吗？
            →→→那个是以下几步：
                ①先从所有金标准提出中，拿出非拥挤矩形的金标准类别ID、外接矩形、掩膜；
                ②计算所有提出和金标准外接矩形重合度，还有他们和拥挤矩形的重合度；
                ③根据各个提出和金标准外接矩形的重合度，选择正例和负例；
                ④下采样，即根据指定的正例和负例个数，去在正例和负例中选择这么多个；
                ⑤把刚刚弄出来的每一个正例，对应到了某个金标准提出上；
                ⑥计算外接矩形修正值、弄对应掩膜什么的。
            目的首先就是不一样的，这个是要弄金标准正例和负例，那是要在已知金标准锚是正例还是负例还是拥挤矩形的情况下，
                来弄RPN提出的正例和负例，然后把正例提出和相应的金标准提出对应起来。
            然后那个是先把锚细化为提出，并且有下采样、弄掩膜的步骤，这个没有。
            还有就是，这儿是弄出是正例还是负例还是中性，而class MaskRCNN里的【从图片到提出的第三步】做了类似事情，得到那个rpn_class_logits，
                与这儿输出的金标准input_rpn_match对比得到损失啊。
            类似的地方是，正例和负例都是根据重合度选择的、都计算了外接矩形修正值（以备后续程序使用）。
        """
    return rpn_match, rpn_bbox

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def get_sparse_gt(h, w, gt_boxes, projection_num):
    row1 = gt_boxes[:, 0]
    row2 = gt_boxes[:, 2]
    col1 = gt_boxes[:, 1]
    col2 = gt_boxes[:, 3]
    sparse_boxes = np.zeros([h, w])  # 图像长宽，到时候改掉。
    for i in range(len(row1)):
        r1 = np.maximum(row1[i], 0)
        r2 = np.minimum(row2[i], h-1) # 防止越界，因为那个sparse_boxes是从0~511的。有点奇怪的是，用原来数据集训练的时候，并没发生过这个问题。
        c1 = np.maximum(col1[i], 0)
        c2 = np.minimum(col2[i], w-1)
        sparse_boxes[r1, c1] = 1
        sparse_boxes[r2, c2] = 1  # 现在只是把左上右下两个角点给投影了，而不是四个。。。所以说，0°的投影轴对应的稀疏编码，应该是有2个非零值。
    theta = np.linspace(0., 180., num=projection_num, endpoint=False)
    sparse_radon = radon(sparse_boxes, theta=theta, circle=True)  # shape是(原图大小, projection_num)，就是在projection_num条直线上的投影结果。然后看了一下，第0条直线，就是投影角度为0的那个，就是原来是1的地方投影为1，而其他的投影直线，一时不知道是怎么搞出来的，可以先了解一下他确实是把金标准往不同直线做了投影，然后再看一下那个radon函数，是怎么确定的投影直线。
    sparse_radon_max = np.max(sparse_radon)  # 试着归一化一下，就是让所有的数都不超过1
    sparse_radon /= sparse_radon_max
    return sparse_radon, sparse_boxes

def batch_processing(process_func, input_batch, **kwargs):
    """现在这个函数，可以处理参数函数返回一个或多个变量的情况了。
    """
    processed = []
    processed_all = []  # 这样，就不需要写一大堆的(参数名_all)这样的变量，然后一个一个append在concatenate了。
    for i in range(input_batch.shape[0]):
        slice = input_batch[i]  # 相当于是input_batch[i,:,:]或者[i,:]或者[i,:,:,:]什么的
        processed = process_func(slice, **kwargs)  # 输入这个func函数的参数，这儿不应该有，而应该通过**kwargs在外面输入
        processed_all.append(processed)
    assert type(processed) in [np.ndarray, tuple], '暂时不支持输出别的类型'
    if isinstance(processed, np.ndarray):  # 【基础】判断是不是np.array。如果是的话，就认为这个函数返回了1个变量。似乎还没考虑返回一个数的情况。。
        processed_all = np.stack(processed_all, axis=0)
        return processed_all
    else:  # 如果不是，就认为这个函数返回的是tuple，也就是多个变量。
        processed_all_zipped = list(zip(*processed_all))
        # 上句，processed_all_zipped是个list，里面的每一个元素都是tuple（这个时候好像不太方便给他变成np.array，毕竟这个函数
        #     不知道process_func的输入输出，也就不知道他有几个元素啊）。
        result = [np.stack(o, axis=0) for o in zip(processed_all_zipped)]
        # 上句，弄成list，然后主函数里可以直接用，见调用的时候。但是调用的时候就会发现，比起单个输出的，就都多了一维。不过，可以在这儿就把它删掉。
        result = [np.squeeze(r) for r in result]  # 删掉多余的维度（是前面的zip等操作搞出来的）
        if len(result) == 1:
            result = result[0]
        return result

def trim_gt(input_gt, num_classes):
    """就是把金标准也弄成10*10的东西，先把补零都删了，然后按照坐标顺序排序。"""
    ix_unpadded = np.where(np.sum(input_gt, axis=1) > 0)[0]  # 非补0的位置
    gt = input_gt[ix_unpadded]
    sorted = gt[gt[:, 0].argsort()]  # 按照第一列（上面的y坐标）排序
    max_class_id = np.max(sorted[:, 4])
    min_class_id = np.min(sorted[:, 4])
    P_before = int(num_classes - 1 - max_class_id)
    P_after = int(min_class_id - 1)
    # padded = np.pad(sorted, ((P_before, P_after), (0, 0)), 'constant', constant_values=(0))
    padded = np.pad(sorted, ((0, P_before+P_after), (0, 0)), 'constant', constant_values=(0))
    return padded

classes = ["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10"]  # 这个是我的检测的json。
import class_detection_dataset
def get_MRCNN_json(loaded_json_file):
    s = loaded_json_file
    images = s['images']
    images = np.array(images)  # 刚才是list，现在变成array，应该是(135, 512, 512)的np.array。
    masks = s['masks']
    masks = np.array(masks)  # 应该也是(135, 512, 512)的np.array。

    all_patients = images.shape[0]  # 一共多少张图。
    dataset = class_detection_dataset.Dataset()  # 构造了一个Dataset类型变量dataset。
    for i, classname in enumerate(classes):
        dataset.add_class("organs", i + 1, classname)
    image_info = {}
    width = images.shape[1]
    height = images.shape[2]
    images_for_dataval = []
    for p in range(all_patients):  # all_patients
        patient_id = p
        image_temp = images[p, :, :]  # 这儿只有1张图了。。
        mask_temp = masks[p, :, :]
        image_temp = image_temp.astype(np.uint8)  # 变成np.uint8【此处忘了图像归一化，还好读进来的图就是给归一化到0~255的，否则就会不对了。。】
        mask_temp = mask_temp.astype(np.uint8)  # 变成np.uint8
        non_zero_ix = mask_temp.nonzero()
        non_zero = mask_temp[non_zero_ix]  # 非零的类别序号。
        classes_num = np.max(non_zero) - np.min(non_zero) + 1 + 1  # 这张图一共多少类。最后的+1是加上背景类。

        mask = np.zeros((width, height, classes_num))
        mask_labeled = np.zeros((width, height, classes_num))  # 把每一类的再分成几块。
        all_organs = 0  # 一开始的器官数
        organ_num = np.zeros(classes_num)  # 每种器官的个数
        for i in range(np.min(non_zero), np.max(non_zero)+1):  # i=0的是背景，就让他是一大堆0好了，然后这个i就是从最小的类别标签到最大的类别标签。
            """此for循环，详见后面的注释。。"""
            i1 = i - np.min(non_zero) + 1  # 从label最小的类别开始，这样，如果没有第1类，那么mask[:, :, 1]就是第2类的所有物体的总掩膜。
            mask[:, :, i1] = mask_temp == i  # 三维矩阵的第三维，这儿用[:,:,i]而MATLAB里是(:,:,i)。
            mask_labeled[:, :, i1] = measure.label(mask[:, :, i1], connectivity=1)  # 这一类（第i类）所有物体，每一个物体的掩膜给一个不同的值。
            m = np.max(mask_labeled[:, :, i1])
            all_organs = all_organs + m  # 一共多少个物体
            organ_num[i1] = m  # 这一类一共多少个物体
        all_organs = all_organs.astype(int)
        mask_detailed = np.zeros((width, height, all_organs + 1))  # 要+1。
        category_detailed = np.zeros(all_organs + 1)
        sum_before = int(1)
        for i in range(1, classes_num):  # 对每个类循环
            num = (organ_num[i]).astype(int)  # 这一类一共多少个物体。这儿是i而不是上面的i1了。
            category_detailed[sum_before:sum_before + num] = i + np.min(non_zero) - 1 # 这一类的物体的label。
            for j in range(sum_before, sum_before + num):  # j是在sum_before~sum_before+num-1之间取值
                mask_detailed[:, :, j] = (mask_labeled[:, :, i] == j - sum_before + 1)
                """
                上句，j在sum_before, sum_before + num之间取值，那就是从1到all_organs之间
                同时，j-sum_before+1在1~num之间取值。如果第i种器官有num个，
                    那这num个肯定都在mask_labeled[:,:,i]这张图里，而且每个的掩膜已经被标记为1~num这些数了。
                验证了一下，上面都是没问题的，其中：
                    1、mask[:, :, 1]~mask[:, :, classes_num-1]（classes_num-1通常是6）是每一种器官（通常共6种）的掩膜；
                    2、mask_labeled[:, :, 1]~mask_labeled[:, :, classes_num-1]是和上面的是一样的，也是每一种器官的掩膜，
                        只不过每一种器官中，“同种而不同个”的器官用了不同的数字去表示（如第一种器官Normal Vertebrae可能有4个，
                        在这个掩膜里就表示为1~4）；
                    3、mask_detailed[:,:,1]~mask_detailed[:,:,all_organs]（all_organs通常是22）是每一个器官的掩膜，
                        此处的第三维就是每一个器官，然后第i个掩膜的器官的种类，就对应相应的category_detailed[i]。
                    4、注意到，mask[:, :, 0]、mask_labeled[:, :, 0]、mask_detailed[:,:,0]，所以：
                        有6种器官，那么mask和mask_labeled就都到[:,:,6]；
                        有22个器官，那么category_detailed和mask_detailed就都到22。
                """
            sum_before = sum_before + num
        category_detailed = category_detailed.astype(int)
        masks_for_dataset = []
        category_for_dataset = []
        for j in range(1, all_organs + 1):  # 对所有的器官循环
            # 器官总个数，一般就是22个。j让他从1~22（而不是从0到21），因为category_detailed的索引是从0到22，而category_detailed[0]是背景。
            classname = classes[category_detailed[j] - 1]  # 本来是classes[category_detailed[j] - 1]现在为了避免保存文字，就不要了
            """
            上句，如果category_detailed[j]=1的话，那么对应的应该是Normal Vertebrae，但是呢，
                这个Normal Vertebrae对应的是classes[0]啊。
            """
            masks_for_dataset.append(mask_detailed[:, :, j].astype(int))
            # j=1的时候，是第1个器官，即应该对应mask[:, :, 1]  试过用.astype(int)转成int，结果他输出的是np.int32，不能用，草他妈的。
            category_for_dataset.append(classname)  # 这个应该是'L4', 'L5', 'S1'那三种器官了。
        images_for_dataval.append(image_temp.astype(np.uint8))  # 先变成uint8再添加到images_for_dataval。
        dataset.add_image(patient_id, image_temp, mask_temp, masks_for_dataset, width, height, category_for_dataset)
    image_info["organs"] = dataset.image_info
    dataset.prepare()
    return dataset

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    从掩膜计算外接矩形。
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([4], dtype=np.int32)
    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]  # 掩膜值为1的地方。随机用一张图验证过，发现np.any(m, axis=0)是一大堆TRUE和FALSE，然后np.where(...axis=0)就是从左到右一列列地去找，找到m中有1的那些列（此例的输出为232,233,234...271），后面的[0]是个小细节，把(?,)变成()。
    vertical_indicies = np.where(np.any(m, axis=1))[0]  # 类似上上句，从上到下一行行地找，找到m中有1的那些行。
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to resizing or cropping. Set bbox to zeros
        # 如果没有掩膜，就把外接矩形全都置零。
        x1, x2, y1, y2 = 0, 0, 0, 0
    boxes= np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)

def to_torch(x):
    x1=cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    x1 = np.transpose(x1, [2,0,1])
    x1=x1/255
    x1=torch.from_numpy(x1)
    x1=x1.to(torch.float32)
    return x1