data = [];
path='../data/chisato/goal/';
data_f = 'log*.csv';
csv_file_num = dir(strcat(path,data_f));

%�w�肵��path+data_f�ɂ���f�[�^��ϐ��udata�Z�v�Ƃ��ă��[�N�X�y�[�X�ɒǉ�
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

%data�Z�ɍs����������
for t=1:length(csv_file_num)
    file_name = csv_file_num(t).name;
    d = strcat('data',num2str(t));
    str1 = [d,'(:,2)'];
    str2 = [d,'(:,1)'];
    plot(eval(str1),eval(str2),'o');
end
hold off;
saveas(gcf,'chisato_goal.png');
