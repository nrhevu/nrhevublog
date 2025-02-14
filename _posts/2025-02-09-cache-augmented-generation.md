---
layout: post
title: "Tạo truy xuất bằng  Cache-Augmented Generation (CAG)"
description: ""
date: 2025-02-09T00:00:00-00:00
tags: Large Language Model
---

# Cache-Augmented Generation (CAG)
Để xây dựng một hệ thống ứng dụng **Large Language Model**, chắc hẳn chúng ta sẽ phải áp dụng một vài kỹ thuật để tăng cường khả năng của mô hình, điển hình nhất là sử dụng những kỹ thuật về tăng cường truy xuất RAG (**Retrieval Augmented Generation**). Vậy trước khi tìm hiểu về CAG, ta hãy xem qua RAG là gì, có điểm mạnh, điểm yếu là gì nhé.

## RAG là gì
RAG là một kỹ thuật giúp cải thiện câu trả lời của các mô hình ngôn ngữ lớn, giảm thiểu tình trạng ảo giác (halluciation) bằng cách thêm những thông tin văn bản liên quan đến câu đầu vào từ một kho văn bản lớn bằng các kỹ thuật tìm kiếm.

Cách thức hoạt động có thể được mô tả như sau:

![alt text](/nrhevublog/images/2025-02-09-cache-augmented-generation/rag.png)

- Đầu tiên, bộ các văn bản chứa tri thức liên quan sẽ được thực hiện đánh chỉ mục (indexing) để phục vụ cho việc tìm kiếm. Trong RAG các kỹ thuật tìm kiếm dày (Dense Retrieval) thường được sử dụng do mục đích tìm kiếm phần lớn sẽ dựa trên ngữ nghĩa của văn bản (documents) và câu hỏi (query). Các mô hình dùng để biểu diễn ẩn (embedding model) văn bản sẽ được sử dụng (có thể tham khảo một vài mô hình tại [mteb leaderboard](https://huggingface.co/spaces/mteb/leaderboard)), các cơ sở dữ liệu vector (vector database) cũng được sử dụng để lưu trữ và truy xuất nhanh văn bản (một vài vector database có thể tham khảo tại [đây](https://superlinked.com/vector-db-comparison))

- Tiếp theo, hệ thống sẽ tìm kiếm những văn bản có liên quan đến câu query của người dùng rồi sẽ thêm dữ kiện trong văn bản vào câu prompt để hỏi LLM, từ đó mô hình sẽ có khả năng hiểu biết tốt hơn và tạo ra câu trả lời tốt hơn.

#### Lợi ích của RAG
- Luôn có thông tin up-to-date mà không cần phải training lại
- Tăng độ chính xác cho những câu hỏi liên quan đến lĩnh vực hẹp (domain-specific)
- Có khả năng diễn giải về nguồn của câu trả lời
- Giảm thiểu tình trạng ảo giác


#### Điểm yếu của RAG
- Làm tăng thời gian tính toán và chi phí về lưu trữ, bảo trì
- Phụ thuộc vào chất lượng mô hình tìm kiếm
- Cài đặt khó khăn và phức tạp


## CAG là gì
Với những điểm yếu đã nêu ở trên, ta cần một phương pháp hiệu quả hơn để cải thiện các thách thức của mô hình ngôn ngữ lớn. Đó là lý do CAG (Cache-Augmented Generation) ra đời. Ý tưởng cơ bản của CAG là sẽ tải trước toàn bộ văn bản vào trong extended context của model và tình toán trước giá trị key-value cache. Với cách tiếp cận này, CAG có thể đảm bảo rằng quá trình suy luận mà có phụ thuộc vào ngữ cảnh sẽ trở nên nhanh và tin cậy hơn. Vậy cụ thể CAG đã làm như nào? Ta cùng đi tìm hiểu tiếp.

## CAG hoạt động như nào
Kỹ thuật "key" nhất của CAG chính là K-V Cache, vậy chúng ta hãy cùng tìm hiểu về Self-Attention và KV cache trước khi xem CAG hoạt động như nào nha.

### Ôn lại về self attention và KV cache
Self-attention là một kỹ thuật siêu nổi tiếng trong lĩnh vực học sâu, cốt lõi của mô hình Transformers, ông tổ của các mô hình LLM bây giờ. Kỹ thuật này cho phép mô hình có thể "tập trung" vào y một vài phần trong chuỗi đầu vào trong quá trình sinh ra token kế tiếp. Ví dụ một câu: "She poured the coffee into the cup" thì mô hình sẽ có thể tập trung hơn vào từ "poured" và từ "coffee" khi sinh từ "into" tiếp theo vì các từ này cung cấp thông tin ngữ cảnh cho từ kế tiếp. 



## So sánh RAG và CAG

## Kết luận

## Tài liệu tham khảo

1. Chan, Brian J., et al. "Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks." arXiv preprint arXiv:2412.15605 (2024).
2. [RAG vs CAG vs Fine-tuning: A Deep Dive into Faster, Smarter Knowledge Integration](https://www.linkedin.com/pulse/rag-vs-cag-fine-tuning-deep-dive-faster-smarter-knowledge-arya-b2nfc/)