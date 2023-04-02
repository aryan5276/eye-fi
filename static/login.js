/* ========================================= * 
		        BEST VIEWED FULLSCREEN
   https://codepen.io/ig_design/full/KKVQpVP
 * ========================================= */

   "use strict";

   const body = document.body;
   const menu = body.querySelector(".menu");
   const menuItems = menu.querySelectorAll(".menu__item");
   const menuBorder = menu.querySelector(".menu__border");
   let activeItem = menu.querySelector(".active");
   
   function clickItem(item) {
     menu.style.removeProperty("--timeOut");
   
     if (activeItem == item) return;
   
     if (activeItem) {
       activeItem.classList.remove("active");
     }
   
     item.classList.add("active");
     activeItem = item;
     offsetMenuBorder(activeItem, menuBorder);
   }
   
   function offsetMenuBorder(element, menuBorder) {
     const offsetActiveItem = element.getBoundingClientRect();
     const left = Math.floor(offsetActiveItem.left - menu.offsetLeft - (menuBorder.offsetWidth - offsetActiveItem.width) / 2) + "px";
     menuBorder.style.transform = `translate3d(${left}, 0 , 0)`;
   }
   
   offsetMenuBorder(activeItem, menuBorder);
   
   menuItems.forEach((item) => {
     item.addEventListener("click", () => clickItem(item));
   });
   
   window.addEventListener("resize", () => {
     offsetMenuBorder(activeItem, menuBorder);
     menu.style.setProperty("--timeOut", "none");
   });
   