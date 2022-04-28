
from preprocessing import *
from models import *
from train_eval import *


data_name = 'ml_1m_stratified'

ratio = 0
rating_map = None
post_rating_map = None
adj_dropout = 0
max_nodes_per_hop = 100
data_appendix ="_mnph10"
epochs = 40
val_test_appendix = "testmode"
transfer = 'results/goodreads_stratified_testmode/'

datasplit_path = (
            'raw_data/' + data_name + '/split_seed' + str(1234) + 
            '.pickle'
        )

if data_name in ['flixster', 'douban', 'yahoo_music']:
    (
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
        test_v_indices, class_values
    ) = load_data_monti(data_name, True, rating_map, post_rating_map)
elif data_name == 'ml_100k':
    print("Using official MovieLens split u1.base/u1.test with 20% validation...")
    (
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
        test_v_indices, class_values
    ) = load_official_trainvaltest_split(
        data_name, True, rating_map, post_rating_map, 1.0
    )
elif data_name == "ml_1m_stratified" or  data_name == "goodreads_stratified":
    print("loading from file without features")
    (
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
        test_v_indices, class_values
        # ,  testtopk_u_indices, testtopk_v_indices
    ) = load_from_file(
        data_name, True, rating_map, post_rating_map, 1.0
    )

else:
    (
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
        test_v_indices, class_values
    ) = create_trainvaltest_split(
        data_name, 1234, True, datasplit_path, True, True, rating_map, 
        post_rating_map, 1.0
    )

# print("loading from file without features")

# (
#     u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
#     val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
#     test_v_indices, class_values
# ) = load_from_file(
#     data_name, True, rating_map, post_rating_map, 1.0
# )

u_features, v_features = None, None
n_features = 0


train_indices = (train_u_indices, train_v_indices)

test_indices = (test_u_indices, test_v_indices)
l = len(test_u_indices)
# testtopk_indices = (testtopk_u_indices, testtopk_v_indices)

# testtopk_indices = (testtopk_u_indices[:l], testtopk_v_indices[:l])



train_graphs, val_graphs, test_graphs = None, None, None

data_combo = (data_name, data_appendix, val_test_appendix)
file_dir = os.path.dirname(os.path.realpath('__file__'))

res_dir = os.path.join(
    file_dir, 'results/{}{}_{}'.format(
        data_name, '', val_test_appendix
    )
)
post_rating_map = {
            x: int(i // (5 / 5)) 
            for i, x in enumerate(np.arange(1, 6).tolist())
        }
# dataset_class = 'MyDynamicDataset' 

dataset_class = 'MyDataset'
print('++++++++++++++++= start herererer')

evaled = eval(dataset_class)
print('done here')
print(evaled)

# train_graphs = eval(dataset_class)(
#     'data/{}{}/{}/train'.format(*data_combo),
#     adj_train, 
#     train_indices, 
#     train_labels, 
#     1, 
#     1.0, 
#     max_nodes_per_hop, 
#     u_features, 
#     v_features, 
#     class_values, 
#     max_num=None,
#     parallel=False
# )


test_graphs = eval(dataset_class)(
    'data/{}{}/{}/test'.format(*data_combo),
    adj_train, 
    test_indices, 
    test_labels, 
    1,
    1.0,
    max_nodes_per_hop, 
    u_features, 
    v_features, 
    class_values, 
    max_num=None,
    parallel=False
)


# test_graphs_topk = eval(dataset_class)(
#     'data/{}{}/{}/testtopk'.format(*data_combo),
#     adj_train, 
#     testtopk_indices, 
#     # test_indices,
#     # test_labels, 
#     np.random.randint(5, size=len(testtopk_indices[0])), 
#     1,
#     1.0,
#     max_nodes_per_hop, 
#     u_features, 
#     v_features, 
#     class_values, 
#     max_num=None,
#     parallel=False
# )
# print(np.random.randint(5, size=len(testtopk_indices)).shape)
# print(len(testtopk_indices[0]))
# # print(len(test_graphs_topk))
# print(len(test_graphs))

num_relations = len(class_values)
multiply_by = 1

model = IGMC (
        test_graphs, 
        latent_dim=[32, 32, 32, 32], 
        num_relations=num_relations, 
        num_bases=4, 
        regression=True, 
        adj_dropout=adj_dropout, 
        force_undirected=False, 
        side_features=False,
        n_side_features=n_features, 
        multiply_by=multiply_by
    )
total_params = sum(p.numel() for param in model.parameters() for p in param)

train_indices = (train_u_indices, train_v_indices)

test_indices = (test_u_indices, test_v_indices)

print(max(train_u_indices), max(train_v_indices), max(test_u_indices), max(test_v_indices))


print('model loaded')

# score = test_once(test_graphs, model, 50, logger=None)
if transfer == '':
    model_pos = os.path.join(res_dir, 'model_checkpoint{}.pth'.format(epochs))
else:
    model_pos = os.path.join(transfer, 'model_checkpoint{}.pth'.format(epochs))
model.load_state_dict(torch.load(model_pos, map_location=torch.device('cpu')))
score = test_once(test_graphs, model, 50, logger=None, evalmethod='rmse')

# score = test_once(test_graphs_topk, model, 50, logger=None, evalmethod='recall', test_graphs = test_graphs)
print(score)
# score = test_once(test_graphs_topk, model, 50, logger=None,  evalmethod='ndcg', test_graphs=test_graphs)
# print(score)

