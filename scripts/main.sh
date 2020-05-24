PYTHON='python'
data_name='mag-acm-full'
# data_names='mag-acm-2000-2000 mag-acm-2001-2001 mag-acm-2002-2002 mag-acm-2003-2003 mag-acm-2004-2004 mag-acm-2005-2005 mag-acm-2006-2006 mag-acm-2007-2007 mag-acm-2008-2008 mag-acm-2009-2009 mag-acm-2010-2010 mag-acm-2011-2011 mag-acm-2012-2012 mag-acm-2013-2013 mag-acm-2014-2014 mag-acm-2015-2015'
# data_names='mag-acm-0-inf'
data_names='the-movies-castcrew-2011-inf the-movies-castkws-2011-inf'
test_ratio=0.1
num_exp=2
for data_name in $data_names
do
    echo $data_name
    echo "Without GCC:"
    $PYTHON ../src/data_preparer.py --data-name $data_name --silent --test-ratio $test_ratio --num-exp $num_exp
    $PYTHON ../src/link_predictor.py --data-name $data_name --silent --num-exp $num_exp > $data_name.out &
done
