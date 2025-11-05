# generate_more_tests.py
from PIL import Image, ImageDraw, ImageFont, ImageFilter

img = Image.new("RGB", (800,500), "white")
d = ImageDraw.Draw(img)
d.text((60,80),"GOVT SCHEME RECEIPT",fill="black")
d.text((60,160),"Name: Rohan Sharma",fill="black")
d.text((60,220),"Amount: Rs. 1500",fill="black")
d.text((60,280),"Date: 12/03/2023",fill="black")
d.text((60,360),"Signature: _________________",fill="black")
img.save("original_doc.jpg")

# forged 1: change amount
f1 = img.copy()
d1 = ImageDraw.Draw(f1)
d1.rectangle((55,210,400,240), fill="white")   # cover amount
d1.text((60,220),"Amount: Rs. 4500", fill="black")
f1.save("forged_amount.jpg")

# forged 2: change name + small blur
f2 = img.copy()
d2 = ImageDraw.Draw(f2)
d2.rectangle((60,150,400,180), fill="white")
d2.text((60,160),"Name: Mohit Kumar", fill="black")
f2 = f2.filter(ImageFilter.GaussianBlur(radius=0.6))
f2.save("forged_name_blur.jpg")

# forged 3: edited and re-saved lower quality
f3 = f1.copy()
f3.save("forged_amount_lowq.jpg", quality=60)
print("Saved 4 files: original_doc.jpg, forged_amount.jpg, forged_name_blur.jpg, forged_amount_lowq.jpg")
