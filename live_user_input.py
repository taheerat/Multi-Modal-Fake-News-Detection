# -*- coding: utf-8 -*-


print("\n📥 Multi-Source Fake News Detection")
user_text = input("Enter news text or URL: ").strip()
img_path = input("Enter image path (optional): ").strip()
vid_path = input("Enter video path (optional): ").strip()

if any(x in user_text for x in ["http://", "https://"]):
    extracted = extract_text_from_url(user_text)
    print("\nExtracted Content:", extracted[:300], "...\n")
    user_text = extracted

input_data = text_transform(user_text)
input_ids = input_data['input_ids'].to(device)
attn_mask = input_data['attention_mask'].to(device)
img_tensor = load_image(img_path).to(device) if img_path else torch.zeros(1, 3, 224, 224).to(device)
vid_tensor = load_video(vid_path).to(device) if vid_path else torch.zeros(1, 3, 16, 112, 112).to(device)

with torch.no_grad():
    logits = model(input_ids, attn_mask, img_tensor, vid_tensor)
    prediction = torch.argmax(logits, dim=1).item()
    print(f"\n✅ Prediction for your input: {'REAL' if prediction == 1 else 'FAKE'}")
