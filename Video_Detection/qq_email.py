import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

my_sender = '2262218068@qq.com'  # 填写发信人的邮箱账号
my_pass = 'ntdoulipufececag'  # 发件人邮箱授权码
my_user = '897783228@qq.com'  # 收件人邮箱账号


def mail():
    ret = True
    try:
        msg = MIMEText('警告，当前发现有人员摔倒，请立即联系家人确认！', 'plain', 'utf-8')  # 填写邮件内容
        msg['From'] = formataddr(["摔倒报警", my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        msg['To'] = formataddr(["test", my_user])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msg['Subject'] = "发生摔倒事件"  # 邮件的主题，也可以说是标题

        server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱授权码
        server.sendmail(my_sender, [my_user, ], msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()  # 关闭连接
    except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
        ret = False
    return ret


if __name__ == '__main__':
    ret = mail()
    if ret:
        print("邮件发送成功")
    else:
        print("邮件发送失败")
