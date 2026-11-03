(async function downloadUnsplashOriginalZip() {
  // 1. JSZip 동적 로드
  if (typeof JSZip === 'undefined') {
    console.log('📦 JSZip 라이브러리를 불러오는 중...');
    await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  const zip = new JSZip();
  const imgFolder = zip.folder("unsplash_originals");

  // 2. Unsplash 이미지 링크 및 원본 URL 추출
  // 페이지 내 모든 a 태그 및 img 태그 중 Unsplash 원본 이미지 연결 URL 탐색
  const rawUrls = [];
  
  // A. 다운로드 버턴/링크가 존재하는 경우 (a[href*="/download"])
  document.querySelectorAll('a[href*="/photos/"][href*="/download"]').forEach(a => {
    if (a.href) rawUrls.push(a.href);
  });

  // B. img 태그 중 images.unsplash.com CDN URL 파싱 및 고화질 파라미터 재구성
  document.querySelectorAll('img[src*="images.unsplash.com"], img[srcset*="images.unsplash.com"]').forEach(img => {
    let src = img.src || img.dataset.src;
    if (!src && img.srcset) {
      const srcsetArr = img.srcset.split(',');
      src = srcsetArr[srcsetArr.length - 1].trim().split(' ')[0];
    }
    
    if (src && src.includes('images.unsplash.com')) {
      try {
        const urlObj = new URL(src);
        // 미리보기, 워터마크 레이어 및 해상도 제한 파라미터 제거
        urlObj.searchParams.delete('w');
        urlObj.searchParams.delete('h');
        urlObj.searchParams.delete('crop');
        urlObj.searchParams.delete('fit');
        urlObj.searchParams.delete('dpr');
        // 고화질 원본 렌더링 파라미터 세팅
        urlObj.searchParams.set('q', '100');
        urlObj.searchParams.set('fm', 'jpg');
        rawUrls.push(urlObj.toString());
      } catch (e) {}
    }
  });

  const uniqueUrls = [...new Set(rawUrls)];

  if (uniqueUrls.length === 0) {
    console.log('❌ 다운로드할 Unsplash 원본 이미지를 찾을 수 없습니다.');
    console.log('💡 스크롤을 끝까지 내려 이미지를 먼저 로드한 뒤 실행해 주세요.');
    return;
  }

  console.log(`🔍 총 ${uniqueUrls.length}개의 원본 이미지 URL 추출 완료. 압축 다운로드를 시작합니다...`);

  // 3. 이미지를 Fetch하여 ZIP에 추가
  let successCount = 0;
  for (let i = 0; i < uniqueUrls.length; i++) {
    const url = uniqueUrls[i];
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const blob = await response.blob();
      const fileName = `unsplash_original_${String(i + 1).padStart(3, '0')}.jpg`;
      
      imgFolder.file(fileName, blob);
      successCount++;
      console.log(`[${i + 1}/${uniqueUrls.length}] 원본 압축 완료: ${fileName}`);
    } catch (err) {
      console.warn(`[${i + 1}/${uniqueUrls.length}] Skip (Fetch 실패/CORS): ${url}`);
    }
  }

  if (successCount === 0) {
    console.error('❌ 이미지를 불러오지 못했습니다.');
    return;
  }

  // 4. ZIP 파일 생성 및 트리거
  console.log('⚙️ 원본 이미지 ZIP 압축 생성 중...');
  const content = await zip.generateAsync({ type: "blob" });

  const a = document.createElement('a');
  a.href = URL.createObjectURL(content);
  a.download = `unsplash_clean_originals_${Date.now()}.zip`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);

  console.log(`✅ 워터마크 없는 고화질 원본 이미지 ${successCount}개 일괄 다운로드 완료!`);
})();