/**
 * HistoCore Documentation - Animation & Interaction Scripts
 * Adds smooth scroll animations, fade-ins, and interactive elements
 */

(function() {
  'use strict';

  // ═══════════════════════════════════════════════════════════════
  // Intersection Observer for Scroll Animations
  // ═══════════════════════════════════════════════════════════════
  
  const observerOptions = {
    threshold: 0.05,
    rootMargin: '0px 0px -20px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-in');
        // Unobserve after animation to improve performance
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  // Observe elements for scroll animations
  function initScrollAnimations() {
    const animatedElements = document.querySelectorAll(`
      section h2,
      section h3,
      section p,
      section ul,
      section ol,
      section table,
      section blockquote,
      .callout
    `);

    animatedElements.forEach((el, index) => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(20px)';
      el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
      el.style.transitionDelay = `${Math.min(index * 0.05, 0.3)}s`;
      observer.observe(el);
    });
  }

  // Add animate-in class styles
  const style = document.createElement('style');
  style.textContent = `
    .animate-in {
      opacity: 1 !important;
      transform: translateY(0) !important;
    }
  `;
  document.head.appendChild(style);

  // ═══════════════════════════════════════════════════════════════
  // Smooth Scroll for Anchor Links
  // ═══════════════════════════════════════════════════════════════
  
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (href === '#') return;
        
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
          target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });
        }
      });
    });
  }

  // ═══════════════════════════════════════════════════════════════
  // Mobile Navigation Toggle
  // ═══════════════════════════════════════════════════════════════
  
  function initMobileNav() {
    const navToggle = document.querySelector('.nav-toggle');
    const nav = document.querySelector('nav#main-nav');
    
    if (navToggle && nav) {
      navToggle.addEventListener('click', () => {
        navToggle.classList.toggle('is-open');
        nav.classList.toggle('nav-open');
      });
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // Code Block Copy Button
  // ═══════════════════════════════════════════════════════════════
  
  function initCodeCopyButtons() {
    document.querySelectorAll('pre').forEach(pre => {
      const button = document.createElement('button');
      button.className = 'copy-code-btn';
      button.textContent = 'Copy';
      button.setAttribute('aria-label', 'Copy code to clipboard');
      
      button.addEventListener('click', async () => {
        const code = pre.querySelector('code');
        const text = code ? code.textContent : pre.textContent;
        
        try {
          await navigator.clipboard.writeText(text);
          button.textContent = 'Copied!';
          button.classList.add('copied');
          
          setTimeout(() => {
            button.textContent = 'Copy';
            button.classList.remove('copied');
          }, 2000);
        } catch (err) {
          button.textContent = 'Failed';
          setTimeout(() => {
            button.textContent = 'Copy';
          }, 2000);
        }
      });
      
      pre.style.position = 'relative';
      pre.appendChild(button);
    });
  }

  // Add copy button styles
  const copyBtnStyle = document.createElement('style');
  copyBtnStyle.textContent = `
    .copy-code-btn {
      position: absolute;
      top: 8px;
      right: 8px;
      padding: 6px 12px;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 4px;
      color: #fff;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      z-index: 10;
    }
    
    .copy-code-btn:hover {
      background: rgba(255, 255, 255, 0.2);
      border-color: rgba(255, 255, 255, 0.3);
    }
    
    .copy-code-btn.copied {
      background: rgba(76, 175, 80, 0.3);
      border-color: rgba(76, 175, 80, 0.5);
    }
  `;
  document.head.appendChild(copyBtnStyle);

  // ═══════════════════════════════════════════════════════════════
  // Progress Bar on Scroll
  // ═══════════════════════════════════════════════════════════════
  
  function initProgressBar() {
    const progressBar = document.createElement('div');
    progressBar.className = 'scroll-progress';
    document.body.appendChild(progressBar);
    
    window.addEventListener('scroll', () => {
      const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = (winScroll / height) * 100;
      progressBar.style.width = scrolled + '%';
    });
  }

  // Add progress bar styles
  const progressStyle = document.createElement('style');
  progressStyle.textContent = `
    .scroll-progress {
      position: fixed;
      top: 0;
      left: 0;
      height: 3px;
      background: linear-gradient(90deg, #A51C30 0%, #8a1728 100%);
      z-index: 9999;
      transition: width 0.1s ease-out;
      box-shadow: 0 2px 4px rgba(165, 28, 48, 0.3);
    }
  `;
  document.head.appendChild(progressStyle);

  // ═══════════════════════════════════════════════════════════════
  // Back to Top Button
  // ═══════════════════════════════════════════════════════════════
  
  function initBackToTop() {
    const button = document.createElement('button');
    button.className = 'back-to-top';
    button.innerHTML = '↑';
    button.setAttribute('aria-label', 'Back to top');
    document.body.appendChild(button);
    
    window.addEventListener('scroll', () => {
      if (window.pageYOffset > 300) {
        button.classList.add('visible');
      } else {
        button.classList.remove('visible');
      }
    });
    
    button.addEventListener('click', () => {
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });
  }

  // Add back to top styles
  const backToTopStyle = document.createElement('style');
  backToTopStyle.textContent = `
    .back-to-top {
      position: fixed;
      bottom: 30px;
      right: 30px;
      width: 50px;
      height: 50px;
      background: linear-gradient(135deg, #A51C30 0%, #8a1728 100%);
      color: white;
      border: none;
      border-radius: 50%;
      font-size: 24px;
      font-weight: bold;
      cursor: pointer;
      opacity: 0;
      visibility: hidden;
      transform: translateY(20px);
      transition: all 0.3s ease;
      box-shadow: 0 4px 12px rgba(165, 28, 48, 0.4);
      z-index: 1000;
    }
    
    .back-to-top.visible {
      opacity: 1;
      visibility: visible;
      transform: translateY(0);
    }
    
    .back-to-top:hover {
      transform: translateY(-4px);
      box-shadow: 0 8px 20px rgba(165, 28, 48, 0.5);
    }
    
    .back-to-top:active {
      transform: translateY(-2px);
    }
    
    @media (max-width: 768px) {
      .back-to-top {
        bottom: 20px;
        right: 20px;
        width: 45px;
        height: 45px;
        font-size: 20px;
      }
    }
  `;
  document.head.appendChild(backToTopStyle);

  // ═══════════════════════════════════════════════════════════════
  // Initialize All Features
  // ═══════════════════════════════════════════════════════════════
  
  function init() {
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }
    
    initScrollAnimations();
    initSmoothScroll();
    initMobileNav();
    initCodeCopyButtons();
    initProgressBar();
    initBackToTop();
    
    console.log('HistoCore documentation animations initialized');
  }

  // Start initialization
  init();

})();
