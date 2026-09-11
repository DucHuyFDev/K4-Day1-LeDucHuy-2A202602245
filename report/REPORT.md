# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/9/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:**
Python: 3.13.15
PyTorch: 2.11.0+cu128
Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): `class_id`: 468, `"class_name"`: "cab", `rank`: 1, `"score"`: 0.510915, `"taxonomy_name"`: "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào? kiểm tra toàn bộ bức ảnh và chọn ra đối tượng nổi bật nhất
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Tập dữ liệu huấn luyện định nghĩa class list. Mô hình chỉ có thể nhận diện các nhãn thuộc các lớp này
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? Tránh sự nhập nhằng, không rõ ràng trong quản lý dữ liệu. ID để máy tính xử lý nhanh, tên lớp để con người đọc dễ hiểu hơn, taxonomy là hệ quy chiếu tương ứng với id nhận dạng
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline cần quy định rõ nguyên tắc chọn nhãn đại diện. Ví dụ: Ưu tiên gán nhãn cho chủ thể chiếm diện tích lớn nhất, chủ thể nằm ở chính giữa ảnh, hoặc chủ thể quan trọng nhất theo mục tiêu nghiệp vụ. Hoặc, guideline có thể yêu cầu chuyển bài toán thành dạng đa nhãn (multi-label) để liệt kê tất cả chủ thể
- Vì sao model score không phải ground truth? Model score chỉ là mức độ tự tin (xác suất tính toán bằng thuật toán) của máy tính khi đưa ra dự đoán. Trong khi đó, ground truth là "sự thật cơ sở" (nhãn chuẩn xác tuyệt đối) do con người gán từ trước để làm thước đo đánh giá máy tính. Dù điểm score của mô hình có cao đến 99% thì dự đoán đó vẫn có thể bị sai lệch so với ground truth.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): `{"class_name": "person", "score": 0.912625, "bbox_xyxy": [385.33, 69.24, 498.92, 348.92], "bbox_width": 113.58, "bbox_height": 279.68}`
- Diễn giải vị trí box bằng lời: Hộp giới hạn có góc trên bên trái nằm tại tọa độ x = 385.33, y = 69.24 và góc dưới bên phải tại x = 498.92, y = 348.92. Hộp có chiều rộng 113.58 pixel và chiều cao 279.68 pixel, bao quanh đối tượng người.
- So sánh số prediction ở hai threshold: thể hiện qua trường "score_threshold": 0.35 và chỉ số "score" của từng record trong file. Nếu tăng threshold lên cao, record có điểm score thấp hơn ngưỡng này sẽ bị loại bỏ, làm giảm số lượng prediction. Ngược lại, mô hình sẽ giữ lại nhiều record hơn, làm tăng số lượng prediction.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Khi hạ threshold, độ bao phủ tăng nhưng reviewer phải xem xét và lọc bỏ nhiều dự đoán sai (box rác) hơn. Ngược lại số lượng box rác giảm giúp reviewer nhàn hơn, nhưng dễ bị sót các vật thể thật có điểm score thấp
- Đề xuất một quy tắc box chặt: Hộp giới hạn phải ôm sát vừa khít phần viền ngoài cùng của vật thể, không để khoảng trống thừa quá lớn và không được cắt lẹm vào bên trong đối tượng
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định rõ ngưỡng tỷ lệ bị che khuất tối đa để quyết định có giữ lại hay không, cách xử lý khi vật thể bị cắt ngang bởi vật cản, và nguyên tắc vẽ box đối với các vật thể bị cắt cụt ở mép khung hình ảnh

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): `{"instance_id": "traffic-001", "class_name": "bus", "score": 0.925745, "polygon_point_count": 120, "polygon_xy": [[148.0, 189.0], [147.0, 190.0], [145.0, 190.0]]}`
- Polygon bổ sung chi tiết gì so với box? tập hợp các điểm tọa độ bám sát viền hình dáng quang học chuẩn xác của đối tượng ở cấp độ pixel, giúp mô hình loại bỏ hoàn toàn các vùng nhiễu (background) và vạch rõ ranh giới mảng khối mà bounding box hình chữ nhật bắt buộc phải chứa đựng bên trong.
- `instance_id` dùng để làm gì và không phải loại ID nào? định danh tính cá biệt và duy nhất cho từng đối tượng vật lý cụ thể xuất hiện trong ảnh nhằm phục vụ bóc tách, tracking độc lập; và nó hoàn toàn không phải là class_id vốn chỉ mang ý nghĩa phân loại nhóm nhãn của đối tượng đó.
- Đề xuất một quy tắc biên mask: Đường biên của mask tạo bởi các điểm tọa độ phải bám khít viền pixel ngoài cùng thuộc về phần thân vật lý của đối tượng, nghiêm ngặt loại bỏ và không được bao bọc phần bóng đổ (shadow), hình phản chiếu hay các hiệu ứng quang học ngoại vi
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Tài liệu nghiệp vụ cần quy định cứng việc mô hình sẽ thực hiện khoanh vùng viền theo pixel thực tế mắt thường nhìn thấy hay phải tự nội suy hình học phần bị che khuất, đồng thời bắt buộc có cơ chế escalation cho admin quyết định cấu hình loại bỏ hoặc gán nhãn thủ công khi đối tượng bị che lấp vượt quá giới hạn hoặc ranh giới tiếp xúc bị mờ nhòe hoàn toàn vào môi trường

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Label/Class ID (String/Integer) gán cho toàn bộ bức ảnh. | Ảnh mờ/nhiễu; ảnh chứa nhiều đối tượng thuộc các class khác nhau (ambiguity); định nghĩa nhãn bị chồng chéo. | Đánh giá tổng thể và chọn nhãn (hoặc tập hợp nhãn) đại diện chính xác nhất cho ngữ nghĩa bức ảnh theo quy định. | Kiểm tra tính chính xác của nhãn được chọn; đối chiếu sự nhất quán với guideline đối với các trường hợp ngoại lệ (edge cases). |
| Phát hiện vật thể | Bounding box `[x_min, y_min, x_max, y_max]` hoặc `[x_center, y_center, width, height]` kèm Class ID. | Box vẽ quá rộng (chứa nhiều nhiễu) hoặc quá hẹp (cắt lẹm vật thể); bỏ sót vật thể nhỏ; vật thể bị che khuất (occlusion) không rõ giới hạn. | Vẽ hộp chữ nhật bám sát nhất vào các điểm cực (trên, dưới, trái, phải) của phần vật thể hiển thị rõ, sau đó gán nhãn. | Đánh giá độ khít của box (thường dựa trên cảm quan IoU); phát hiện vật thể bị bỏ sót (False Negative), dán nhầm class, kiểm tra quy tắc xử lý vùng che khuất. |
| Instance segmentation | Mảng tọa độ Polygon `[[x1, y1], [x2, y2], ...]` hoặc Pixel Mask kèm Class ID và Instance ID. | Điểm polygon lẹm vào nền (background); bao gồm cả bóng đổ (shadow)/hình phản chiếu; các instance đè lên nhau không được tách rời rõ ràng. | Chấm các điểm tọa độ bám sát đường viền vật lý ở cấp độ pixel của từng đối tượng riêng biệt; gán định danh duy nhất (instance_id). | Kiểm tra độ chuẩn xác của viền mask ở mức pixel; đánh giá khả năng tách biệt các instance chồng chéo; xác nhận đã loại trừ bóng và các phần nhiễu ngoại vi theo quy tắc biên. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Tuyệt đối không sao chép, tải xuống hoặc chia sẻ dữ liệu nội bộ ra bên ngoài dưới mọi hình thức.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Quản trị viên hệ thống (Admin) hoặc Quản lý trực tiếp.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
