import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
import ssl

def send_mail(send_from, send_to, subject, text, file_path=None,
              server="smtp.gmail.com", port=465, app_password=None):
    assert isinstance(send_to, list)

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    if file_path:
        with open(file_path, "rb") as fil:
            part = MIMEApplication(fil.read(), Name=basename(file_path))
            part['Content-Disposition'] = f'attachment; filename="{basename(file_path)}"'
            msg.attach(part)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(server, port, context=context) as server:
            server.login(send_from, app_password)
            server.sendmail(send_from, send_to, msg.as_string())
            print("Email sent successfully.")
    except Exception as e:
        print(f"Error: {e}")
