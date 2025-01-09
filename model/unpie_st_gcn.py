import tensorflow as tf

from utils.graph_utils import normalize_undigraph

"""The basic module for applying a spatial graph convolution.
    Args:
        filters (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, C, T, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size
            :math:`K` is the spatial kernel size
            :math:`T` is a length of the sequence
            :math:`V` is the number of graph nodes
            :math:`C` is the number of incoming channels
"""
class SGCN(tf.keras.Model):
    def __init__(self, 
                 channel_out, 
                 dropout_conv,
                 params,
                 channel_in):
        super(SGCN, self).__init__()
        self.conv = tf.keras.layers.Conv2D(channel_out,
                                           kernel_size= (1, 1),
                                           strides = (1, 1),
                                           dilation_rate = (1, 1),
                                           padding='same',
                                           data_format='channels_first',)
        self.drop = tf.keras.layers.Dropout(dropout_conv)

        self.channel_in = channel_in
        self.seq_len = params['seq_len']
        self.num_nodes = params['num_nodes']
        self.batch_size = params['batch_size']

    def call(self, inputs, training):
        x, a = inputs
        # x: N, C, T, V
        # a: N, T, V, V
        x = self.drop(self.conv(x), training)
        # x = self.conv(x)
        x = tf.einsum('nctv,ntvw->nctw', x, a)
        return x, a
    
    def build_graph(self):
        inputs = [
            tf.keras.Input(shape=(self.channel_in, self.seq_len, self.num_nodes), name='x', batch_size=self.batch_size),
            tf.keras.Input(shape=(self.seq_len, self.num_nodes, self.num_nodes), name='a', batch_size=self.batch_size)
        ]

        return tf.keras.Model(inputs, self.call(inputs, training=False))


"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        filters (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        activation (activation function/name, optional): activation function to use
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
        downsample (bool, optional): If ``True``, applies a downsampling residual mechanism. Default: ``True``
                                     the value is used only when residual is ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
"""
class STGCN(tf.keras.Model):
    def __init__(self, 
                 channel_out, 
                 kernel_size,
                 use_mdn,
                 stride,
                 residual,
                 dropout_conv,
                 dropout_tcn,
                 downsample,
                 params,
                 channel_in):
        super(STGCN, self).__init__()
        
        self.use_mdn = use_mdn
        
        self.sgcn = SGCN(channel_out, dropout_conv, params, channel_in)

        self.tgcn = tf.keras.Sequential()
        self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))
        self.tgcn.add(tf.keras.layers.PReLU())
        self.tgcn.add(tf.keras.layers.Conv2D(channel_out,
                                             kernel_size=(kernel_size[0], 1),
                                             strides=(stride, 1),
                                             padding='same',
                                             data_format='channels_first',))
        self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))
        self.tgcn.add(tf.keras.layers.Dropout(dropout_tcn))

        self.act = tf.keras.layers.PReLU()

        if not residual:
            self.residual = lambda x, training: 0
        elif residual and stride == 1 and not downsample:
            self.residual = lambda x, training: x
        else:
            self.residual = tf.keras.Sequential()
            self.residual.add(tf.keras.layers.Conv2D(channel_out,
                                                     kernel_size=(1, 1),
                                                     strides=(stride, 1),
                                                     padding='same',
                                                     data_format='channels_first',))
            self.residual.add(tf.keras.layers.BatchNormalization(axis=1))

        self.seq_len = params['seq_len']
        self.num_nodes = params['num_nodes']
        self.batch_size = params['batch_size']
        self.channel_in = channel_in

    def call(self, inputs, training):
        x, a = inputs
        # x: N, C, T, V
        # a: N, T, V, V

        res = self.residual(x, training)
        x, a = self.sgcn(inputs, training)
        x = self.tgcn(x, training) + res

        if not self.use_mdn:
            x = self.act(x)

        return x, a
    
    def build_graph(self):
        inputs = [
            tf.keras.Input(shape=(self.channel_in, self.seq_len, self.num_nodes), name='x', batch_size=self.batch_size),
            tf.keras.Input(shape=(self.seq_len, self.num_nodes, self.num_nodes), name='a', batch_size=self.batch_size)
        ]

        return tf.keras.Model(inputs, self.call(inputs, training=False))


"""Spatial temporal graph convolutional networks.
    Args:
        num_class (int): Number of classes for the classification task
    Shape:
        - Input: :math:`(N, T_{in}, V_{in}, in_channels)`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes.
"""
class UnPIESTGCN(tf.keras.Model):
    transform = normalize_undigraph

    def __init__(self, **params):
        super(UnPIESTGCN, self).__init__()

        gcn_num_input_layers = params['gcn_num_input_layers']
        gcn_num_middle_layers = params['gcn_num_middle_layers']
        gcn_num_middle_2_layers = params['gcn_num_middle_2_layers']
        gcn_num_output_layers = params['gcn_num_output_layers']
        gcn_input_layer_dim = params['gcn_input_layer_dim']
        gcn_middle_layer_dim = params['gcn_middle_layer_dim']
        gcn_middle_2_layer_dim = params['gcn_middle_2_layer_dim']
        gcn_output_layer_dim = params['gcn_output_layer_dim']
        scene_input_layer_dim = params['scene_input_layer_dim']
        scene_output_layer_dim = params['scene_output_layer_dim']
        scene_num_input_layers = params['scene_num_input_layers']
        scene_num_output_layers = params['scene_num_output_layers']
        drop_tcn = params['drop_tcn']
        drop_conv = params['drop_conv']
        num_nodes = params['num_nodes']
        seq_len = params['seq_len']
        stgcn_kernel_size = params['stgcn_kernel_size']
        self.edge_importance = params['edge_importance']
        self.is_scene = params['is_scene']
        self.share_edge_importance = params['share_edge_importance']

        self.seq_len = params['seq_len']
        self.num_nodes = params['num_nodes']
        self.output_feat_extr = params['feat_output_size']
        self.len_one_hot_classes = params['len_one_hot_classes']
        self.batch_size = params['batch_size']
        self.emb_dim = params['emb_dim']

        self.data_bn_x = tf.keras.layers.BatchNormalization(axis=1)
        if self.is_scene:
            self.data_bn_b = tf.keras.layers.BatchNormalization(axis=1)

        self.STGCN_layers_x = []
        for i in range(gcn_num_input_layers):
            self.STGCN_layers_x.append(
                STGCN(
                    channel_out=gcn_input_layer_dim, 
                    kernel_size=[stgcn_kernel_size, seq_len], 
                    residual=False, 
                    stride=1,
                    use_mdn=True, 
                    dropout_tcn=0.5, 
                    dropout_conv=0.5,
                    downsample=True if i==0 else False,
                    params=params,
                    channel_in=self.output_feat_extr + self.len_one_hot_classes))
        for i in range(gcn_num_output_layers):
            self.STGCN_layers_x.append(
                STGCN(
                    channel_out=gcn_output_layer_dim, 
                    kernel_size=[stgcn_kernel_size, seq_len], 
                    residual=True, 
                    stride=1,
                    use_mdn=True, 
                    dropout_tcn=0.5, 
                    dropout_conv=0.2,
                    downsample=True if i==0 and gcn_output_layer_dim != gcn_input_layer_dim else False,
                    params=params,
                    channel_in=gcn_input_layer_dim))

        if self.is_scene:
            self.STGCN_layers_b = []
            for _ in range(scene_num_input_layers):
                self.STGCN_layers_b.append(
                    STGCN(
                        channel_out=scene_input_layer_dim, 
                        kernel_size=[stgcn_kernel_size, seq_len], 
                        residual=False, 
                        stride=1,
                        use_mdn=True, 
                        dropout_tcn=0.5, 
                        dropout_conv=0,
                        downsample=True if i==0 else False,
                        params=params,
                        channel_in=self.output_feat_extr + self.len_one_hot_classes))
            for _ in range(scene_num_output_layers):
                self.STGCN_layers_b.append(
                    STGCN(
                        channel_out=scene_output_layer_dim, 
                        kernel_size=[stgcn_kernel_size, seq_len], 
                        residual=True, 
                        stride=1,
                        use_mdn=True, 
                        dropout_tcn=0.5, 
                        dropout_conv=0.2,
                        downsample=True if i==0 and scene_output_layer_dim != scene_input_layer_dim else False,
                        params=params,
                        channel_in=scene_input_layer_dim))

        if self.edge_importance:
            # Initialize edge_importance as a list of trainable variables
            self.edge_importance_x = [
                tf.Variable(
                    tf.random.normal(
                        shape=(num_nodes, num_nodes),
                        stddev=tf.sqrt(2.0 / (num_nodes * num_nodes)),  # He initialization
                    ),
                    trainable=True,
                )
                for _ in self.STGCN_layers_x
            ]
            if not self.share_edge_importance and self.is_scene:
                    self.edge_importance_b = [
                        tf.Variable(
                            tf.random.normal(
                                shape=(num_nodes, num_nodes),
                                stddev=tf.sqrt(2.0 / (num_nodes * num_nodes)),  # He initialization
                            ),
                            trainable=True,
                        )
                        for _ in self.STGCN_layers_b
                    ]

    def call(self, inputs, training):
        x, b, c, a = inputs
        # x: N, T, V, C
        # b: N, T, V, 4
        # c: N, T, V, Cl
        # a: N, T, V, V

        # Feature concatenation
        if self.is_scene:
            # Concatenate [x, c] and [b, c]
            x = tf.concat([x, c], axis=-1)
            b = tf.concat([b, c], axis=-1)
        else:
            # Concatenate [x, b, c]
            x = tf.concat([x, b, c], axis=-1)

        # Features branch
        NX, TX, VX, CX = x.shape
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        x = tf.reshape(x, [NX, VX * CX, TX])
        x = self.data_bn_x(x, training)
        x = tf.reshape(x, [NX, VX, CX, TX])
        x = tf.transpose(x, perm=[0, 1, 3, 2])
        x = tf.reshape(x, [NX, CX, TX, VX])

        if self.edge_importance:
            for layer, importance in zip(self.STGCN_layers_x, self.edge_importance_x):
                inputs = (x, a * importance)
                x, _ = layer(inputs, training)
        else:
            for layer in self.STGCN_layers_x:
                inputs = (x, a)
                x, _ = layer(inputs, training)

        x = x[:, :, :, 0] # N,CX,T, get the first node V (pedestrian) for each graph

        if self.is_scene:
            # Scene branch
            NB, TB, VB, CB = b.shape
            b = tf.transpose(b, perm=[0, 2, 3, 1])
            b = tf.reshape(b, [NB, VB * CB, TB])
            b = self.data_bn_b(b, training)
            b = tf.reshape(b, [NB, VB, CB, TB])
            b = tf.transpose(b, perm=[0, 1, 3, 2])
            b = tf.reshape(b, [NB, CB, TB, VB])

            if self.edge_importance:
                self.edge_importance_b = self.edge_importance_x if self.share_edge_importance else self.edge_importance_b
                for layer, importance in zip(self.STGCN_layers_b, self.edge_importance_b):
                    inputs = (b, a * importance)
                    b, _ = layer(inputs, training)
            else:
                for layer in self.STGCN_layers_b:
                    inputs = (b, a)
                    b, _ = layer(inputs, training)

            b = b[:, :, :, 0] # N,CB,T, get the first node V (pedestrian) for each graph
            x = tf.concat([x, b], axis=1) # N,CX+CB,T

        x = tf.transpose(x, perm=[0, 2, 1]) # N,T,CX+CB

        return x

    def build_graph(self):
        inputs = [
            tf.keras.Input(shape=(self.seq_len, self.num_nodes, self.output_feat_extr), name='x', batch_size=self.batch_size),
            tf.keras.Input(shape=(self.seq_len, self.num_nodes, 4), name='b', batch_size=self.batch_size),
            tf.keras.Input(shape=(self.seq_len, self.num_nodes, self.len_one_hot_classes), name='c', batch_size=self.batch_size),
            tf.keras.Input(shape=(self.seq_len, self.num_nodes, self.num_nodes), name='a', batch_size=self.batch_size)
        ]

        return tf.keras.Model(inputs, self.call(inputs, training=False))