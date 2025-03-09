import ssl
import smtplib
import aiosmtplib
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv(override=True)

def send_email(subject, message, recipient_email, sender_email, sender_app_pass, sender_host, sender_port=465):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        context = ssl.create_default_context()

        if sender_port == 465:  # SSL connection
            with smtplib.SMTP_SSL(sender_host, sender_port, context=context) as server:
                server.login(sender_email, sender_app_pass)
                server.sendmail(sender_email, recipient_email, msg.as_string())
        else:  # STARTTLS connection (port 587)
            with smtplib.SMTP(sender_host, sender_port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(sender_email, sender_app_pass)
                server.sendmail(sender_email, recipient_email, msg.as_string())

        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
        raise

async def send_email_async(subject, message, recipient_email, sender_email, sender_app_pass, sender_host, sender_port=465):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        context = ssl.create_default_context()
        if sender_port == 465:  # SSL connection
            await aiosmtplib.send(
                msg,
                hostname=sender_host,
                port=sender_port,
                username=sender_email,
                password=sender_app_pass,
                use_tls=True,
                tls_context=context,
            )
        else:  # STARTTLS connection (port 587)
            await aiosmtplib.send(
                msg,
                hostname=sender_host,
                port=sender_port,
                username=sender_email,
                password=sender_app_pass,
                start_tls=True,
                tls_context=context,
            )

        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
        raise

send_email(
    "Test Subject", "Test Body",
    "husein.70821@hotmail.com", "hussein.70821@gmail.com", "cvlopilatdeifsie",
    "smtp.gmail.com", sender_port=587
)
