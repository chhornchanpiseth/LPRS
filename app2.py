
from flask import Flask, request, jsonify
import easyocr


# reader = easyocr.Reader(['en']) 


# Initialize the EasyOCR reader
reader = easyocr.Reader(['ja'],recog_network='japanese_g2',)
stringList = "'-0123456789あいうえおかきくけこさしすせそたちつてとなにぬねのはひふ〜へほまみむめもやゆよらりるれろをわ北土見斎岡福浦島岐部阜群馬広金沢高知新潟市帯大分徳富山宇都宮札幌仙台沼津千葉梨多摩阪横浜前橋練和歌神戸崎埼玉熊谷品川奈良足立京姫路野田鹿児"

app = Flask(__name__)

@app.route('/api/license_plate', methods=['POST'])
def license_plate():
    image_file = request.files['image']
    image = image_file.read()
    result = reader.readtext(image,paragraph=False,detail = 0, allowlist=stringList)

    return jsonify(result)

if __name__ == '__main__':
    app.run()

