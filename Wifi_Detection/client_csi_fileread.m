clc
clear
% 这个文件就可以实现乒乓操作，服务器端不用改的
% echotcpip("on",10002);   %开端口
my_tcp = tcpclient('47.113.200.64', 7000, 'Timeout', 60,'ConnectTimeout',30);%连接这个ip和这个端口的tcp服务器，后面两个参数都是超时时间，具体可以看文档
% 定义一个数据读取的回调函数，将回调函数设置为每次接收10字节数据时触发。
%configureCallback(t,"byte",10,@readDataFcn);  没用到这个
% 发送数据

i = 1;
flag = 0;
%现在在切换文件的时候，也不会造成数据读不上了，因为远程要等发送的有数据时才能读出
disp('正在发送数据')
while(1)
    if(flag == 0)
        m = matfile("csi_0.mat");   % 交替打开两个文件，实现乒乓操作
        name = whos(m).name;
        rx = m.(name);  % rx就是那个元胞了
    else
        m = matfile("csi_1.mat");
        name = whos(m).name;
        rx = m.(name);  % rx就是那个元胞了
    end
    while(i<=length(rx))
        send_phase = rx{i,1}.CSI.Phase;  % 相位信息
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%下面是发送相位，不用动了%%%%%%%%%%%%%%%%%%%%
        s_phase = num2str(send_phase);
        s_pt = regexprep(s_phase,'\s*',',');   %这可是一个好函数，可以将字符串用一个逗号分隔
        s_pt_len = strlength(s_pt);
        switch strlength(string(s_pt_len))
            case 3
                s_pt_len_str = strcat('00000',string(s_pt_len));
            case 4
                s_pt_len_str = strcat('0000',string(s_pt_len));  %将它固定成8个字节，就是8个字符的形式，后面看情况改
           
        end
        write(my_tcp,s_pt_len_str,"string");     %先发送长度进行通知，固定8个字节长度
        % b(i) = strlength(s_pt);
        write(my_tcp,s_pt, "string");               %再接收当前长度的内容即可
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%幅度信息%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        send_mag = rx{i,1}.CSI.Mag;   % 幅度数据读取

        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%发送幅度信息，不用动了%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        s_mag = num2str(send_mag);
        s_mt = regexprep(s_mag,'\s*',',');   %这可是一个好函数，可以将字符串用一个逗号分隔
        s_mt_len = strlength(s_mt);
        switch strlength(string(s_mt_len))
            case 3
                s_mt_len_str = strcat('00000',string(s_mt_len));
            case 4
                s_mt_len_str = strcat('0000',string(s_mt_len));  %将它固定成8个字节，就是8个字符的形式，后面看情况改
               
        end
        write(my_tcp,s_mt_len_str,"string");     %先发送长度进行通知，固定8个字节长度
        %%a(i) = strlength(s_mt);
        write(my_tcp,s_mt, "string");             %再接收当前长度的内容即可
        
        %该计数用于控制行访问
        i = i+1;
    end
    i = 1;
    if(flag == 0)  % 改变标志位，读取另一个文件
        flag = 1;
        clear m;
        clear name;
        clear rx;
    else
        flag = 0;
        clear m;
        clear name;
        clear rx
    end

end


echotcpip("off");
clear my_tcp
disp('发送成功')