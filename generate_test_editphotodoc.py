from PIL import Image, ImageDraw, ImageFont

# Create Original Document
img = Image.new("RGB", (800, 500), color=(255, 255, 255))
draw = ImageDraw.Draw(img)
draw.text((100, 100), "GOVT. SCHEME CLAIM RECEIPT", fill=(0, 0, 0))
draw.text((100, 200), "Name: Rohan Sharma", fill=(0, 0, 0))
draw.text((100, 250), "Amount: Rs. 1500", fill=(0, 0, 0))
draw.text((100, 300), "Date: 12/03/2023", fill=(0, 0, 0))
draw.text((100, 400), "Signature: _______________", fill=(0, 0, 0))
img.save("original_doc.jpg")
print("✅ Original document saved as 'original_doc.jpg'")

# Create Forged Document (simulate an edited amount)
img_edit = img.copy()
draw2 = ImageDraw.Draw(img_edit)
draw2.rectangle((95, 240, 350, 270), fill=(255, 255, 255))  # cover old amount
draw2.text((100, 250), "Amount: Rs. 4500", fill=(0, 0, 0))  # edited value
img_edit.save("forged_doc.jpg")
print("🚨 Forged document saved as 'forged_doc.jpg'")
