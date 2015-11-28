#! /bin/zsh -
zmodload zsh/zutil
zparseopts -A ARGUMENTS l: f: i: r:

learning_rate=$ARGUMENTS[-l]
iterations=$ARGUMENTS[-i]
filename=$ARGUMENTS[-f]
where_to_test=$ARGUMENTS[-r]

# clean up previous output
rm -rf output
rm thetas.txt
touch thetas.txt

for i in `seq 1 $iterations`;
    do
        echo "iteration " $i  
        python map_reduce.py $filename -q -r $where_to_test --file thetas.txt > output
        python update_thetas.py -l $learning_rate
done
