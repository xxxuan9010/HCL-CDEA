clear all

% 读取Excel文件中的所有数据到数组中
filename = 'data.xlsx';
[num, txt, raw] = xlsread(filename);

% 提取数值数据
data = num;

% 计算全包络关系
for i = 1:size(data, 2)
    data(:,i) = data(:,i) / max(data(:,i));
end

% 提取X和Y矩阵
X = data(:, 3:6)';
Y = data(:, 1:2)';
n_total = size(data, 1);

d1=1;
d2=0;
d3=0;

% [blm_dmu,new_dmu_X,new_dmu_Y] = blmsf(X,Y,d1,d2,d3,1);

% 创建 Excel 文件保存路径
output_file = 'completEnvelope24.xlsx';
sheet_idx = 1;

writematrix([X',Y'], output_file, 'Sheet', 'features');

% 记录有效 DMU（效率为1）
efficient_dmus = [];

for i = 1:n_total
    [blm_dmu, ~, ~] = blmsf(X, Y, d1, d2, d3, i);
    
    if isempty(blm_dmu)
        % DMU i 是有效的
        efficient_dmus = [efficient_dmus; i];
        continue;
    end
    
    % 输出每条超边到 Excel 的一个 sheet
    % 每行格式：[DMU_index, weight, node_1, node_2, ...]
    output = blm_dmu(:,2:end);


    % 将该 DMU 的结果写入 Excel 的独立 sheet（以DMU编号命名）
    writematrix(output, output_file, 'Sheet', num2str(i));
end

% 保存有效DMU编号
% writematrix(efficient_dmus, output_file, 'Sheet', 'EfficientDMUs');

disp('已完成所有 DMU 的包络关系计算与保存。');
system('shutdown –s -t 60')