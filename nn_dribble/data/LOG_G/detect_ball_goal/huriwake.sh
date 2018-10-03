for file in `find . -maxdepth 1 |grep .csv`;
do
	echo $file
	if grep 'error' $file >/dev/null;then
		echo "exsit"
		mv $file ./fall/
	else
		echo "not found"
		mv $file ./stable/
	fi
done
