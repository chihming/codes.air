#
Compilation
```
make
```

## EdgeList.tsv as Input
```
nodeA   nodeB   5
nodeB   nodeC   3
nodeC   nodeA   4
```

## Usage
directly run the execution file to see the usage, for instance:
```
./air
```

## Representation.tsv as Outputs
```
nodeA   0.123 0.321 0.456 0.765
nodeB   0.100 0.270 0.500 0.200
nodeC   0.174 0.124 0.190 0.147
```

## Details
`air` needs 3 types of input graphs:
- `-train_ui`
    - the major interactions we'd like to model
    - cannot be empty file
- `-train_up`
    - the user-profile tsv files
    - able to consider multiple files, seperated by `,`
    - if there is no related files, it shall reads an empty file
- `-train_im`
    - the item-meta tsv files
    - able to consider multiple files, seperated by `,`
    - if there is no related files, it shall reads an empty file

and outputs 2 embedding files:
- `-save_q`
    - as query embedding
- `-save_k`
    - as target embedding

## Examples
Suppose we have
- user-item.view.graph
- user-item.cart.graph
- user-word.search.graph
- item-word.title.graph
- item-word.description.graph
- item-tag.category.graph
- empty.graph

### case 1. consider only user-item relations:
```
./air -train_ui user-item.view.graph -train_up empty.graph -train_im empty.graph -dimension 100 -update_times 100 -worker 4 -save_q miso.q.rep -save_k miso.k.rep
```

### case 2. consider item meta:
```
./air -train_ui user-item.view.graph -train_up empty.graph -train_im item-word.title.graph,item-word.description.graph,item-tag.category.graph -dimension 100 -update_times 100 -worker 4 -save_q miso.q.rep -save_k miso.k.rep
```

### case 3. consider both user profile and item meta:
```
./air -train_ui user-item.view.graph -train_up user-word.search.graph -train_im item-word.title.graph,item-word.description.graph,item-tag.category.graph -dimension 100 -update_times 100 -worker 4 -save_q miso.q.rep -save_k miso.k.rep
```

Please contact CM (changecandy@gmail.com) if you encounter any problem.
