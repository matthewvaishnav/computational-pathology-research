# GitHub Pages Responsive Layout Fix - Bugfix Design

## Overview

This bugfix addresses the responsive layout issue in the GitHub Pages site at viewport widths between 961px and 1200px. The current implementation uses fixed sidebar width (340px) and fixed content margins (320px), creating a cramped content area at medium screen sizes. The fix introduces an intermediate responsive breakpoint with proportionally adjusted layout dimensions to ensure smooth transitions and maintain minimum 600px content width for readability.

The fix strategy involves adding a new CSS media query for the 961px-1200px range that reduces the sidebar width to 280px and adjusts content margins accordingly, providing a smooth transition between full desktop layout (>1200px) and mobile layout (≤960px).

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when viewport width is between 961px and 1200px, causing cramped layout with insufficient content area
- **Property (P)**: The desired behavior for medium viewports - proportionally adjusted sidebar and content widths maintaining minimum 600px content area
- **Preservation**: Existing desktop layout (>1200px), mobile layout (≤960px), navigation functionality, accessibility features, and print styles that must remain unchanged
- **Viewport Width**: The width of the browser window in pixels, used to determine which responsive layout to apply
- **Sidebar (header)**: The left navigation panel with fixed width of 340px in desktop layout, containing site logo, navigation links, and footer
- **Content Section (section)**: The main content area with `margin-left: 320px` and `width: calc(100% - 320px)` in desktop layout
- **Responsive Breakpoint**: A viewport width threshold where CSS media queries trigger layout changes (currently 960px and 480px)
- **Intermediate Breakpoint**: The new 961px-1200px range where proportional layout adjustments will be applied

## Bug Details

### Bug Condition

The bug manifests when a user resizes their browser window to viewport widths between 961px and 1200px. At these medium screen sizes, the CSS maintains the full desktop layout with fixed sidebar width (340px) and fixed content margin (320px), but the total viewport width is insufficient to provide adequate content display area. This results in a cramped layout where the content area is only 640px-880px wide, making the page feel squeezed and poorly adapted.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type ViewportState
  OUTPUT: boolean
  
  RETURN input.viewportWidth >= 961
         AND input.viewportWidth <= 1200
         AND input.sidebarWidth == 340
         AND input.contentMarginLeft == 320
         AND input.contentWidth == (input.viewportWidth - 320)
         AND input.contentWidth < 880
END FUNCTION
```

### Examples

- **Example 1**: Viewport width = 1000px → Sidebar = 340px, Content area = 680px (cramped, should be wider)
- **Example 2**: Viewport width = 1100px → Sidebar = 340px, Content area = 780px (still cramped, should be wider)
- **Example 3**: Viewport width = 960px → Abrupt switch to mobile layout (no smooth transition)
- **Edge Case**: Viewport width = 961px → Just above breakpoint, uses desktop layout but extremely cramped (641px content area)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Desktop layout at viewport widths greater than 1200px must continue to display with 340px sidebar and proper content spacing
- Mobile layout at viewport widths 960px or less must continue to display with full-width stacked content and collapsible navigation
- Navigation functionality, link behavior, and interactive elements must continue to work identically
- Accessibility features (keyboard navigation, focus indicators, reduced motion support) must remain unchanged
- Print styles must continue to apply for optimal printed output
- Touch-friendly controls on mobile devices (≤480px) must remain unchanged

**Scope:**
All inputs that do NOT involve viewport widths between 961px and 1200px should be completely unaffected by this fix. This includes:
- Large desktop viewports (>1200px) - full desktop layout
- Tablet and mobile viewports (≤960px) - mobile layout
- Print media - print-specific styles
- User interactions with navigation, links, and content
- Accessibility features and keyboard navigation

## Hypothesized Root Cause

Based on the bug description and CSS analysis, the root cause is:

1. **Missing Intermediate Breakpoint**: The CSS only has breakpoints at 960px and 480px, with no intermediate responsive adaptation for the 961px-1200px range. This leaves a "dead zone" where the desktop layout is applied but the viewport is too narrow.

2. **Fixed Sidebar Width**: The sidebar uses a fixed width of 340px regardless of viewport size above 960px. This is appropriate for large screens but too wide for medium screens.

3. **Fixed Content Margins**: The content section uses fixed `margin-left: 320px` and `width: calc(100% - 320px)`, which doesn't adapt to medium screen sizes.

4. **Abrupt Transition**: The 960px breakpoint switches directly from desktop to mobile layout without any intermediate states, causing a jarring visual transition.

## Correctness Properties

Property 1: Bug Condition - Proportional Layout at Medium Viewports

_For any_ viewport width between 961px and 1200px (where the bug condition holds), the fixed CSS SHALL apply proportionally reduced sidebar width (280px) and adjusted content margins (260px), ensuring the content area maintains at least 600px width for readability and smooth visual adaptation.

**Validates: Requirements 2.1, 2.3, 2.4**

Property 2: Preservation - Desktop and Mobile Layout Behavior

_For any_ viewport width that is NOT between 961px and 1200px (where the bug condition does NOT hold), the fixed CSS SHALL produce exactly the same layout as the original CSS, preserving full desktop layout (>1200px) with 340px sidebar, mobile layout (≤960px) with stacked content, and all navigation, accessibility, and print functionality.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `docs/assets/css/style.scss`

**Location**: After the existing `@media (max-width: 960px)` block and before the `@media (max-width: 480px)` block

**Specific Changes**:

1. **Add New Media Query Block**: Insert a new `@media (min-width: 961px) and (max-width: 1200px)` media query to target the intermediate viewport range

2. **Reduce Sidebar Width**: Change `header` width from 340px to 280px within the new media query
   - This provides 60px more space for content while maintaining sidebar functionality
   - Sidebar remains functional with navigation and branding visible

3. **Adjust Content Margin**: Change `section` margin-left from 320px to 260px within the new media query
   - Matches the reduced sidebar width (280px) with 20px gap
   - Ensures content doesn't overlap with sidebar

4. **Adjust Content Width**: Change `section` width from `calc(100% - 320px)` to `calc(100% - 260px)` within the new media query
   - Provides proportionally wider content area
   - At 1000px viewport: 740px content area (vs 680px before)
   - At 1100px viewport: 840px content area (vs 780px before)
   - At 1200px viewport: 940px content area (vs 880px before)

5. **Adjust Sidebar Padding**: Reduce `header` padding from `30px 20px 20px` to `25px 15px 15px` within the new media query
   - Compensates for reduced sidebar width
   - Maintains visual balance and readability

6. **Adjust Content Padding**: Reduce `section` padding from `50px 80px 60px` to `40px 50px 50px` within the new media query
   - Provides more usable content width
   - Maintains comfortable reading experience

**Implementation Code**:
```scss
/* ─── Medium screens (961px-1200px) - Intermediate responsive breakpoint ─── */
@media (min-width: 961px) and (max-width: 1200px) {
  header {
    width: 280px;
    padding: 25px 15px 15px;
  }
  
  section {
    margin-left: 260px;
    width: calc(100% - 260px);
    padding: 40px 50px 50px;
  }
}
```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code by testing at various viewport widths in the 961px-1200px range, then verify the fix works correctly and preserves existing behavior at other viewport widths.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Manually test the unfixed site at various viewport widths using browser developer tools. Measure the actual content area width and assess visual cramping. Document specific viewport widths where the layout appears cramped or poorly adapted.

**Test Cases**:
1. **Viewport 1000px Test**: Resize browser to 1000px width, measure content area (will show ~680px, cramped on unfixed code)
2. **Viewport 1100px Test**: Resize browser to 1100px width, measure content area (will show ~780px, still cramped on unfixed code)
3. **Viewport 961px Test**: Resize browser to 961px width, measure content area (will show ~641px, extremely cramped on unfixed code)
4. **Viewport 1200px Test**: Resize browser to 1200px width, measure content area (will show ~880px, approaching acceptable on unfixed code)
5. **Transition Test**: Slowly resize from 1201px to 960px, observe abrupt layout shift at 960px (will show jarring transition on unfixed code)

**Expected Counterexamples**:
- Content area width is less than 700px at 1000px viewport (cramped layout)
- Content area width is less than 800px at 1100px viewport (still cramped)
- Abrupt visual shift when crossing 960px breakpoint (no smooth transition)
- Possible causes: missing intermediate breakpoint, fixed sidebar width, fixed content margins

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds (viewport widths 961px-1200px), the fixed CSS produces the expected behavior (proportionally adjusted layout with adequate content area).

**Pseudocode:**
```
FOR ALL viewportWidth WHERE isBugCondition(viewportWidth) DO
  result := applyFixedCSS(viewportWidth)
  ASSERT result.sidebarWidth == 280
  ASSERT result.contentMarginLeft == 260
  ASSERT result.contentWidth == (viewportWidth - 260)
  ASSERT result.contentWidth >= 700  // Minimum acceptable at 961px
  ASSERT result.contentWidth <= 940  // Maximum at 1200px
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold (viewport widths outside 961px-1200px), the fixed CSS produces the same result as the original CSS.

**Pseudocode:**
```
FOR ALL viewportWidth WHERE NOT isBugCondition(viewportWidth) DO
  ASSERT applyOriginalCSS(viewportWidth) = applyFixedCSS(viewportWidth)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain (viewport widths)
- It catches edge cases that manual unit tests might miss (boundary conditions at 960px, 1200px, 1201px)
- It provides strong guarantees that behavior is unchanged for all non-buggy viewport widths

**Test Plan**: Observe behavior on UNFIXED code first at various viewport widths outside the bug range, then write tests capturing that behavior and verify it continues after fix.

**Test Cases**:
1. **Desktop Layout Preservation (>1200px)**: Test at 1201px, 1400px, 1920px - verify sidebar is 340px, content margin is 320px
2. **Mobile Layout Preservation (≤960px)**: Test at 960px, 768px, 480px, 320px - verify full-width stacked layout with collapsible navigation
3. **Navigation Preservation**: Click all navigation links at various viewport widths - verify links work identically
4. **Accessibility Preservation**: Test keyboard navigation (Tab, Enter) at various viewport widths - verify focus indicators and navigation work
5. **Print Preservation**: Trigger print preview - verify print-specific styles apply correctly
6. **Animation Preservation**: Test with `prefers-reduced-motion` enabled - verify animations are disabled

### Unit Tests

- Test CSS media query activation at boundary conditions (960px, 961px, 1200px, 1201px)
- Test sidebar width calculation at various viewport widths in bug range (961px, 1000px, 1100px, 1200px)
- Test content area width calculation at various viewport widths in bug range
- Test that content area width is always >= 700px in bug range (minimum readability threshold)
- Test that layout transitions smoothly without horizontal scrolling

### Property-Based Tests

- Generate random viewport widths in range 961-1200px and verify sidebar is 280px, content margin is 260px
- Generate random viewport widths outside range 961-1200px and verify layout matches original CSS
- Generate random viewport widths across full range 320-1920px and verify no horizontal scrolling occurs
- Test that content area width is always proportional to viewport width with correct offsets

### Integration Tests

- Test full page rendering at viewport widths: 320px, 480px, 768px, 960px, 961px, 1000px, 1100px, 1200px, 1201px, 1920px
- Test smooth resizing from 1920px down to 320px, verifying layout adapts at each breakpoint
- Test navigation functionality at each breakpoint (click links, verify page navigation)
- Test visual regression by comparing screenshots at each breakpoint before and after fix (should match except 961-1200px range)
- Test accessibility features (keyboard navigation, focus indicators) at each breakpoint
- Test print rendering to verify print styles are unaffected
