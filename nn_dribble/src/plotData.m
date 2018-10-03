data = [];
path='../data/chisato/goal/';
data_f = 'log*.csv';
csv_file_num = dir(strcat(path,data_f));

%指定したpath+data_fにあるデータを変数「data〇」としてワークスペースに追加
for i=1:length(csv_file_num)
    file_name = csv_file_num(i).name;
    file_name = strcat(path,file_name);
    t_data = csvread(file_name);
    str = ['data', num2str(i), ' = t_data'];
    eval(str);
    disp(str);
end

figure(1);
hold on;
grid on;
xlim([-400 400])
ylim([100 700])

%data〇に行いたい処理
for t=1:length(csv_file_num)
    file_name = csv_file_num(t).name;
    d = strcat('data',num2str(t));
    str1 = [d,'(:,2)'];
    str2 = [d,'(:,1)'];
    plot(eval(str1),eval(str2),'o');
end
hold off;
saveas(gcf,'chisato_goal.png');
