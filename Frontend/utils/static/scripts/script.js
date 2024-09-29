function toggleMenu() {
    var menu = document.getElementById('menu');
    var menuIcon = document.getElementById('menu-icon');

    if (menu.classList.contains('visible')) {
        menu.classList.remove('visible');
        menuIcon.textContent = 'M'; // 打开图标
        menuIcon.style.transform = 'rotate(0deg)'; // 重置图标旋转
    } else {
        menu.classList.add('visible');
        menuIcon.textContent = 'M'; // 关闭图标
        menuIcon.style.transform = 'rotate(90deg)'; // 旋转图标
    }
}

function showImage(index) {
    var images = document.querySelectorAll('.slider-image');
    var activeImage = document.querySelector('.slider-image.active');

    if (activeImage) {
        activeImage.classList.add('exit'); // 添加滑出动画
        activeImage.classList.remove('active'); // 移除active类
    }

    images[index].classList.add('active');
    images[index].classList.remove('exit'); // 移除滑出动画类
}

var currentIndex = 0;
var images = document.querySelectorAll('.slider-image');

function nextImage() {
    currentIndex = (currentIndex + 1) % images.length;
    showImage(currentIndex);
}

setInterval(nextImage, 5000);

// Initialize the first image
showImage(currentIndex);

// 新增的部分：用于section2的内容轮换和按钮切换功能
let elementIndex = 0;
const elements = document.querySelectorAll('.element1');
const leftControl = document.getElementById('control-left');
const rightControl = document.getElementById('control-right');

function showElement(index) {
    elements.forEach((element, i) => {
        if (i === index) {
            element.classList.add('active');
        } else {
            element.classList.remove('active');
        }
    });
}

function nextElement() {
    elementIndex = (elementIndex + 1) % elements.length;
    showElement(elementIndex);
}

function prevElement() {
    elementIndex = (elementIndex - 1 + elements.length) % elements.length;
    showElement(elementIndex);
}

leftControl.addEventListener('click', prevElement);
rightControl.addEventListener('click', nextElement);

// 每20秒自动切换元素
setInterval(nextElement, 20000);

// 初始化显示第一个元素
showElement(elementIndex);

function isElementInViewport(el) {
    var rect = el.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

function handleScroll() {
    var elements = document.querySelectorAll('.animated');
    elements.forEach(function(element) {
        if (isElementInViewport(element)) {
            element.classList.add('animate-on-scroll');
        } else {
            element.classList.remove('animate-on-scroll');
        }
    });
}

window.addEventListener('scroll', handleScroll);
handleScroll(); // 初始检查

// 当用户滚动时，显示或隐藏返回顶端按钮
window.onscroll = function() {
    toggleBackToTopButton();
};

function toggleBackToTopButton() {
    var button = document.getElementById("back-to-top");
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        if (button.style.display !== "block") {
            button.style.display = "block";
            setTimeout(function() {
                button.classList.add("show");
            }, 10); // 延迟10ms以确保display属性已经生效
        }
    } else {
        button.classList.remove("show");
        setTimeout(function() {
            if (!button.classList.contains("show")) {
                button.style.display = "none";
            }
        }, 500); // 延迟500ms以确保渐隐动画完成
    }
}

function scrollToTop() {
    document.body.scrollTop = 0; // 对于Safari
    document.documentElement.scrollTop = 0; // 对于Chrome, Firefox, IE和Opera
}